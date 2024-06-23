
#include "chess/chess.h"
#include "dataset/batchloader.h"
#include "dataset/dataset.h"
#include "dataset/io.h"
#include "dataset/process.h"
#include "misc/csv.h"
#include "misc/timer.h"
#include "nn/nn.h"
#include "operations/operations.h"
#include "dataset/binpackloader.h"

#include <fstream>
#include <algorithm>

using namespace nn;
using namespace data;

/*
int PSQTValues[5] = {126, 781, 825, 1276, 2538};

data::DenseMatrix<float> values {0, 0};
binpack::chess::Color side;

data::DenseMatrix<float> PSQT();
data::DenseMatrix<float> getinitialPSQTValues();
*/

struct ChessModelBinpack : nn::Model {
    int   current_epoch = 0;
    int   max_epochs    = 0;

    // seting inputs
    virtual void setup_inputs_and_outputs(binpackloader::DataSet& positions) = 0;

    // train function
    void train(binpackloader::BinpackLoader& train_loader,
               binpackloader::BinpackLoader& val_loader,
               int                            epochs       = 1000,
               int                            train_epoch_size   = 1e8,
               int                            val_epoch_size = 1e7) {
        this->compile(train_loader.batch_size);

        max_epochs = epochs;

        Timer t {};
        for (int i = 1; i <= epochs; i++) {
            t.start();

            current_epoch             = i;

            uint64_t prev_print_tm    = 0;
            float    total_epoch_loss = 0;
            float    total_val_loss   = 0;

            // Training phase
            for (int b = 1; b <= train_epoch_size / train_loader.batch_size; b++) {
                auto ds = train_loader.next();
                setup_inputs_and_outputs(ds);

                float batch_loss = batch();
                total_epoch_loss += batch_loss;
                float epoch_loss = total_epoch_loss / b;

                t.stop();
                uint64_t elapsed = t.elapsed();
                if (elapsed - prev_print_tm > 1000 || b == train_epoch_size / train_loader.batch_size) {
                    prev_print_tm = elapsed;

                    printf("\rep/ba = [%3d/%5d], ", i, b);
                    printf("batch_loss = [%1.8f], ", batch_loss);
                    printf("epoch_loss = [%1.8f], ", epoch_loss);
                    printf("speed = [%7.2f it/s], ", 1000.0f * b / elapsed);
                    printf("time = [%3ds]", (int) (elapsed / 1000.0f));
                    
                    /*
                    * printf("\rep = [%4d], epoch_loss = [%1.8f], batch = [%5d], batch_loss = [%1.8f], "
                           "speed = [%7.2f it/s], time = [%3ds]",
                           i,
                           epoch_loss,
                           b,
                           batch_loss,
                           1000.0f * b / elapsed,
                           (int) (elapsed / 1000.0f));
                    */

                    std::cout << std::flush;
                }
            }

            // Validation phase
            for (int b = 1; b <= val_epoch_size / val_loader.batch_size; b++) {
                auto ds = val_loader.next();
                setup_inputs_and_outputs(ds);

                float val_batch_loss = loss();
                total_val_loss += val_batch_loss;
            }

            float epoch_loss = total_epoch_loss / (train_epoch_size / train_loader.batch_size);
            float val_loss   = total_val_loss / (val_epoch_size / val_loader.batch_size);

            printf(", val_loss = [%1.8f]", val_loss);
            next_epoch(epoch_loss, val_loss);
            std::cout << std::endl;
        }
    }
};

struct SeraphinaModel : ChessModelBinpack {
    static constexpr int THREADS = 16;    // threads to use on the cpu

    SparseInput*         in1;
    SparseInput*         in2;

    SeraphinaModel() : ChessModelBinpack() {
        in1      = add<SparseInput>(32 * 12 * 64, 32);
        in2      = add<SparseInput>(32 * 12 * 64, 32);

        /*
        auto ftpsqt = add<FeatureTransformer>(in1, in2, 8);
        ftpsqt->weights.values = getinitialPSQTValues();
        auto ftpsqtl = add<Linear>(ftpsqt);
        auto ftpsqtout = add<Affine>(ftpsqtl, 1);
        */

        // auto psqtws = add<WeightedSum>(ftpsqt->out_1, ftpsqt->out_2, (int)side - 0.5, -((int)side - 0.5));

        auto ft  = add<FeatureTransformer>(in1, in2, 1536);
        auto ftc = add<ClippedRelu>(ft);
        // ft->ft_regularization = 1.0 / 16384.0 / 4194304.0;
        ftc->max  = 127.0;

        auto l1t = add<Affine>(ftc, 16);
        auto l1d = add<Affine>(l1t, 1);
        auto l1   = add<Affine>(l1t, 15);
        auto l1sc  = add<SqrClippedRelu>(l1);
        auto l1c  = add<ClippedRelu>(l1);

        std::memcpy(l1sc + 15, l1c, 15 * sizeof(std::uint8_t));

        auto l2  = add<Affine>(l1sc, 32);
        auto l2c = add<ClippedRelu>(l2);

        auto l3  = add<Affine>(l2c, 1);
        auto l3c = add<WeightedSum>(l3, l1d, 1.0, 1.0);
        // auto l3temp = add<Affine>(l3c, 1);
        // auto l3t = add<WeightedSum>(l3temp, ftpsqtout, 1.0, 1.0);
        // auto l3s = add<Sigmoid>(l3c, 1.0 / 160.0);

        set_loss(MPE {2.5, true});
        set_lr_schedule(StepDecayLRSchedule {4.375e-4, 0.995, 1});
        const float hidden_max = 127.0 / 64.0;
        add_optimizer(Adam(
            {{OptimizerEntry {&ft->weights}},
             {OptimizerEntry {&ft->bias}},
             {OptimizerEntry {&l1->weights}.clamp(-hidden_max, hidden_max)},
             {OptimizerEntry {&l1->bias}},
             {OptimizerEntry {&l2->weights}.clamp(-hidden_max, hidden_max)},
             {OptimizerEntry {&l2->bias}},
             {OptimizerEntry {&l3->weights}.clamp(-(127 * 127) / 9600, (127 * 127) / 9600)},
             {OptimizerEntry {&l3->bias}}},
            0.9,
            0.999,
            1e-7));

        set_file_output("D:\\Grapheus-Seraphina\\cmake-build-release\\Release\\nnue");
        add_quantization(Quantizer {
            "quant_1",
            10,
            QuantizerEntry<int16_t>(&ft->weights.values, 64.0, true),
            QuantizerEntry<int16_t>(&ft->bias.values, 64.0),
            QuantizerEntry<int8_t>(&l1->weights.values, 64.0),
            QuantizerEntry<int32_t>(&l1->bias.values, 64.0),
            QuantizerEntry<int8_t>(&l2->weights.values, 64.0),
            QuantizerEntry<int32_t>(&l2->bias.values, 64.0),
            QuantizerEntry<int8_t>(&l3->weights.values, 1.0),
            QuantizerEntry<int32_t>(&l3->bias.values, 64.0),
        });
        set_save_frequency(10);
    }

    static int king_square_index(chess::Square relative_king_square) {

        // clang-format off
        constexpr int indices[chess::N_SQUARES] {
            -1, -1, -1, -1, 31, 30, 29, 28,
            -1, -1, -1, -1, 27, 26, 25, 24,
            -1, -1, -1, -1, 23, 22, 21, 20,
            -1, -1, -1, -1, 19, 18, 17, 16,
            -1, -1, -1, -1, 15, 14, 13, 12,
            -1, -1, -1, -1, 11, 10, 9, 8,
            -1, -1, -1, -1, 7, 6, 5, 4,
            -1, -1, -1, -1, 3, 2, 1, 0,
        };
        // clang-format on

        return indices[relative_king_square];
    }

    static int index(chess::Square piece_square,
                     chess::Piece  piece,
                     chess::Square king_square,
                     chess::Color  view) {

        const chess::PieceType piece_type  = chess::type_of(piece);
        const chess::Color     piece_color = chess::color_of(piece);

        piece_square ^= 56;
        king_square ^= 56;

        const int oP  = piece_type + 6 * (piece_color != view);
        const int oK  = (7 * !(king_square & 4)) ^ (56 * view) ^ king_square;
        const int oSq = (7 * !(king_square & 4)) ^ (56 * view) ^ piece_square;

        return king_square_index(oK) * 12 * 64 + oP * 64 + oSq;
    }

    /*
    void getside(binpackloader::DataSet& positions) {
        for (int b = 0; b < positions.size(); b++) {
            const auto entry = positions[b];
            side             = entry.pos.sideToMove();
        }
    }
    */

    void setup_inputs_and_outputs(binpackloader::DataSet& positions) {
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        auto&             target               = m_loss->target;

#pragma omp parallel for schedule(static) num_threads(16)
        for (int b = 0; b < positions.size(); b++) {
            const auto       entry = positions[b];

            binpackloader::BinpackLoader::set_features(b, entry, in1, in2, index);

            float p_value  = binpackloader::BinpackLoader::get_p_value(positions[b]);
            float w_value  = binpackloader::BinpackLoader::get_w_value(positions[b]);

            // float lambda   = 1.0;

            float p        = (p_value - 270) / 380;
            float pm       = (-p_value - 270) / 380;
            float p_target = 0.5 * (1.0 + 1 / (1 + expf(-p)) - 1 / (1 + expf(-pm)));
            // float p_target = 1 / (1 + expf(-p_value * 1.0 / 410.0));

            float w        = (w_value - 270) / 340;
            float wm       = (-w_value - 270) / 340;
            float w_target = 0.5 * (1.0 + 1 / (1 + expf(-w)) - 1 / (1 + expf(-wm)));
            // float w_target = (w_value + 1) / 2.0f;

            float actual_lambda = 1.0 + (1.0 - 0.7) * (current_epoch / max_epochs);

            target(b) = (actual_lambda * p_target + (1.0f - actual_lambda) * w_target) / 1.0f;
            // target(b) = lambda * p_target + (1.0 - lambda) * w_target;
        }
    }
};

/*
// Piece Square Table
data::DenseMatrix<float> PSQT()
{
    int indexw = 0;
    int indexb = 0;

    for (int ks = 0; ks < 64; ks++)
    {
        for (int s = 0; s < 64; s++)
        {
            for (int pv : PSQTValues)
            {
                if (pv == PSQTValues[0])
                {
                    indexw = SeraphinaModel::index(s, chess::PAWN, ks, chess::WHITE);
                    indexb = SeraphinaModel::index(s, chess::PAWN, ks, chess::BLACK);
                }

                if (pv == PSQTValues[1])
                {
                    indexw = SeraphinaModel::index(s, chess::KNIGHT, ks, chess::WHITE);
                    indexb = SeraphinaModel::index(s, chess::KNIGHT, ks, chess::BLACK);
                }

                if (pv == PSQTValues[2])
                {
                    indexw = SeraphinaModel::index(s, chess::BISHOP, ks, chess::WHITE);
                    indexb = SeraphinaModel::index(s, chess::BISHOP, ks, chess::BLACK);
                }

                if (pv == PSQTValues[3])
                                {
                                        indexw = SeraphinaModel::index(s, chess::ROOK, ks,
chess::WHITE); indexb = SeraphinaModel::index(s, chess::ROOK, ks, chess::BLACK);
                                }

                if (pv == PSQTValues[4])
                {
                    indexw = SeraphinaModel::index(s, chess::QUEEN, ks, chess::WHITE);
                                        indexb = SeraphinaModel::index(s, chess::QUEEN, ks,
chess::BLACK);
                }

                values[indexw] = pv;
                values[indexb] = -pv;
            }
        }
    }

    return values;
}

data::DenseMatrix<float> getinitialPSQTValues()
{
    return (PSQT() + 64 * 12) / 600;
}
*/

int main() {
    init();

    // Fetch training dataset paths
    std::string tf = "E:/trainingdata/S4/leela96-dfrc99-v2-T78juntosepT79mayT80junsepnovjan-v6dd-T80mar23-v6-T60novdecT77decT78aprmayT79aprT80may23.min.binpack";
    std::vector<std::string> train_file = {tf};

    // Fetch validation dataset paths
    std::string vf = "E:/trainingdata/S4/leela96-dfrc99-v2-T78juntosepT79mayT80junsepnovjan-v6dd-T80mar23-v6-T60novdecT77decT78aprmayT79aprT80may23.min.binpack";
    std::vector<std::string> val_file = {vf};

    binpackloader::BinpackLoader train_loader {train_file, 16384, 16};
    train_loader.start();

    binpackloader::BinpackLoader val_loader {val_file, 16384, 16};
    val_loader.start();

   SeraphinaModel model {};
    //    model.train(loader, 1000, 1e8);
//    model.distribution(loader, 32);

//    model.test_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//    model.compile(16384);
//    model.setup_inputs_and_outputs(loader.next());
//    model.batch();
//    std::cout << model.loss_of_last_batch() << std::endl;



//    model.load_weights("../res/run1/weights/test.state");
//    model.test_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//    model.train(loader, 1000, 1e8);
    model.load_weights("D:/Grapheus-Seraphina/cmake-build-release/Release/nnue/weights/S3.state");
    // model.distribution(loader, 84);
//    model.quantize("Nyx.nnue");

//    model.test_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//    model.save_weights("../res/run1/weights/test.state");

//    model.compile(1);
//    model.load_weights(R"(C:\Users\Luecx\CLionProjects\Grapheus\res\run1\weights\300.state)");
//    model.test_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//
//    model.load_weights(R"(C:\Users\Luecx\CLionProjects\Grapheus\res\run1\weights\200.state)");
//    model.test_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");


    model.train(train_loader, val_loader, 800, 1e8, 1e7);

    close();
    return 0;
}
