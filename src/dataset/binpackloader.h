#pragma on

// turn off warnings for this
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "../binpack/nnue_data_binpack_format.h"
#pragma GCC diagnostic pop

#include "../nn/layers/input.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>

namespace binpackloader {

using DataEntry = binpack::binpack::TrainingDataEntry;
using DataSet   = std::vector<DataEntry>;
using binpack::binpack::CompressedTrainingDataEntryParallelReader;

// Data Filtering Strategy taken from Stockfish nnue pytorch
// https://github.com/official-stockfish/nnue-pytorch/blob/master/training_data_loader.cpp
std::function<bool(const DataEntry&)> make_skip_predicate(bool filtered,
                                                          int  random_fen_skipping,
                                                          bool wld_filtered,
                                                          int  early_fen_skipping) {
    if (filtered || random_fen_skipping || wld_filtered || early_fen_skipping) {
        return [random_fen_skipping,
                prob = static_cast<double>(random_fen_skipping) / (random_fen_skipping + 1),
                filtered,
                wld_filtered,
                early_fen_skipping](const DataEntry& e) {
            static constexpr int VALUE_NONE  = 32002;

            auto                 do_wld_skip = [&]() {
                std::bernoulli_distribution distrib(1.0
                                                    - e.score_result_prob() * e.score_result_prob());
                auto&                       prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_skip = [&]() {
                std::bernoulli_distribution distrib(prob);
                auto&                       prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_filter = [&]() {
                return e.isInCheck() || (e.isCapturingMove() && (e.score == 0 || e.seeGE(0)));
            };

            // Allow for predermined filtering without the need to remove positions from the dataset.
            if (e.score == VALUE_NONE)
                return true;

            if (e.ply <= early_fen_skipping) {
                return true;
            }

            if (random_fen_skipping && do_skip()) {
                return true;
            }

            if (filtered && do_filter())
                return true;

            if (wld_filtered && do_wld_skip())
                return true;

            return false;
        };
    }

    return nullptr;
}

/// @brief Multithreaded dataloader to load data in Stockfish's binpack format
struct BinpackLoader {
    using DATAENTRY_TYPE = binpack::binpack::TrainingDataEntry;
    using DATASET_TYPE   = std::vector<DataEntry>;

    static constexpr std::size_t                               ChunkSize = (1 << 22);

    std::vector<std::string>                                   paths;
    std::unique_ptr<CompressedTrainingDataEntryParallelReader> reader;

    std::vector<std::size_t>                                   permute_shuffle;
    DataSet                                                    buffer;
    DataSet                                                    active_buffer;
    DataSet                                                    active_batch;

    std::thread                                                readingThread;
    int                                                        batch_size;
    int                                                        current_batch_index  = 0;
    size_t                                                     total_positions_read = 0;
    int                                                        concurrency          = 8;

    static constexpr auto                 openmode = std::ios::in | std::ios::binary;

    std::function<bool(const DataEntry&)> skipPredicate;

    BinpackLoader(const std::vector<std::string>& filename,
                  int                             batch_size,
                  int                             concurrency,
                  int                             early_fen_skipping,
                  int                             random_fen_skipping)
        : batch_size(batch_size)
        , paths(filename)
        , concurrency(concurrency) {
        buffer.reserve(ChunkSize);
        active_buffer.reserve(ChunkSize);
        permute_shuffle.resize(ChunkSize);
        active_batch.reserve(batch_size);

        skipPredicate = make_skip_predicate(true, random_fen_skipping, true, early_fen_skipping);
        reader        = std::make_unique<binpack::binpack::CompressedTrainingDataEntryParallelReader>(
            concurrency,
            paths,
            openmode,
            false,
            skipPredicate);
    }

    void start() {

        current_batch_index = 0;

        shuffle();
        loadNext();
        loadToActiveBuffer();
        readingThread = std::thread(&BinpackLoader::loadNext, this);
    }

    void loadToActiveBuffer() {
        active_buffer.clear();
        for (int i = 0; i < buffer.size(); i++) {
            active_buffer.push_back(buffer[i]);
        }
    }

    void loadNext() {
        buffer.clear();

        auto k = reader->fill(buffer, ChunkSize);

        if (ChunkSize != k) {
            reader = std::make_unique<binpack::binpack::CompressedTrainingDataEntryParallelReader>(
                concurrency,
                paths,
                openmode,
                false,
                skipPredicate);
        }
    }

    DataSet& next() {
        active_batch.clear();

        for (int i = 0; i < batch_size; i++) {
            if (current_batch_index >= active_buffer.size()) {

                current_batch_index = 0;

                if (readingThread.joinable()) {
                    readingThread.join();
                }

                loadToActiveBuffer();
                shuffle();

                readingThread = std::thread(&BinpackLoader::loadNext, this);
            }

            active_batch.push_back(active_buffer[permute_shuffle[current_batch_index++]]);
        }

        return active_batch;
    }

    void shuffle() {
        std::iota(permute_shuffle.begin(), permute_shuffle.end(), 0);
        std::shuffle(permute_shuffle.begin(),
                     permute_shuffle.end(),
                     std::mt19937(std::random_device()()));
    }

    static auto loadFen(const std::string& fen) {
        binpack::chess::Position pos;
        pos.set(fen);

        binpackloader::DataEntry entry;
        entry.pos    = pos;
        entry.score  = 0;
        entry.result = 0;

        return entry;
    }

    static auto get_p_value(const DataEntry& entry) {
        return entry.score;
    }

    static auto get_w_value(const DataEntry& entry) {
        return entry.result;
    }

    template<typename InputIndexFunction>
    static void set_features(const int          batch,
                             const DataEntry&   entry,
                             nn::SparseInput*   in1,
                             nn::SparseInput*   in2,
                             InputIndexFunction index) {
        const auto& pos     = entry.pos;
        const auto  wKingSq = pos.kingSquare(binpack::chess::Color::White);
        const auto  bKingSq = pos.kingSquare(binpack::chess::Color::Black);

        const auto  pieces  = pos.piecesBB();

        for (auto sq : pieces) {
            const auto         piece                 = pos.pieceAt(sq);
            const std::uint8_t pieceType             = static_cast<uint8_t>(piece.type());
            const std::uint8_t pieceColor            = static_cast<uint8_t>(piece.color());

            auto               piece_index_white_pov = index(static_cast<int>(sq),
                                               pieceType,
                                               static_cast<int>(wKingSq),
                                               static_cast<uint8_t>(binpack::chess::Color::White));
            auto               piece_index_black_pov = index(static_cast<int>(sq),
                                               pieceType,
                                               static_cast<int>(bKingSq),
                                               static_cast<uint8_t>(binpack::chess::Color::Black));

            if (pos.sideToMove() == binpack::chess::Color::White) {
                in1->sparse_output.set(batch, piece_index_white_pov);
                in2->sparse_output.set(batch, piece_index_black_pov);
            } else {
                in2->sparse_output.set(batch, piece_index_white_pov);
                in1->sparse_output.set(batch, piece_index_black_pov);
            }
        }
    }
};

}    // namespace binpackloader