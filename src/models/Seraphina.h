#pragma once

#include "chessmodel.h"

namespace model {

struct SeraphinaModel : ChessModel<binpackloader::BinpackLoader> {
    using DataLoader = binpackloader::BinpackLoader;

    SparseInput* in1;
    SparseInput* in2;

    const float  sigmoid_scale = 1.0 / 160.0;
    const float  quant_one     = 64.0;
    const float  quant_two     = 64.0;

    const size_t n_features    = 32 * 12 * 64;
    const size_t n_l1          = 16;
    const size_t n_l2          = 32;
    const size_t n_out         = 1;

    float        start_lambda;
    float        end_lambda;

    SeraphinaModel(binpackloader::BinpackLoader&                train_loader,
                   std::optional<binpackloader::BinpackLoader>& val_loader,
                   int                                          n_ft,
                   float                                        start_lambda,
                   float                                        end_lambda,
                   int                                          save_rate)
        : start_lambda(start_lambda)
        , end_lambda(end_lambda)
        , ChessModel(train_loader, val_loader) {

        in1                    = add<SparseInput>(n_features, 32);
        in2                    = add<SparseInput>(n_features, 32);

        auto ft                = add<FeatureTransformer>(in1, in2, n_ft);
        auto fta               = add<ClippedRelu>(ft);
        ft->ft_regularization  = 1.0 / 16384.0 / 4194304.0;
        fta->max               = 127.0;

        auto        l1         = add<Affine>(fta, n_l1);
        auto        l1a        = add<ClippedRelu>(l1);

        auto        l2         = add<Affine>(l1a, n_l2);
        auto        l2a        = add<ClippedRelu>(l2);

        auto        pos_eval   = add<Affine>(l2a, n_out);
        auto        sigmoid    = add<Sigmoid>(pos_eval, sigmoid_scale);

        const float hidden_max = 127.0 / quant_two;
        add_optimizer(AdamWarmup({{OptimizerEntry {&ft->weights}},
                                  {OptimizerEntry {&ft->bias}},
                                  {OptimizerEntry {&l1->weights}.clamp(-hidden_max, hidden_max)},
                                  {OptimizerEntry {&l1->bias}},
                                  {OptimizerEntry {&l2->weights}.clamp(-hidden_max, hidden_max)},
                                  {OptimizerEntry {&l2->bias}},
                                  {OptimizerEntry {&pos_eval->weights}},
                                  {OptimizerEntry {&pos_eval->bias}}},
                                 0.95,
                                 0.999,
                                 1e-8,
                                 5 * 16384));

        set_save_frequency(save_rate);
        add_quantization(Quantizer {
            "quant",
            (size_t)save_rate,
            QuantizerEntry<int16_t>(&ft->weights.values, quant_one, true),
            QuantizerEntry<int16_t>(&ft->bias.values, quant_one),
            QuantizerEntry<int8_t>(&l1->weights.values, quant_two),
            QuantizerEntry<int32_t>(&l1->bias.values, quant_two),
            QuantizerEntry<int8_t>(&l2->weights.values, quant_two),
            QuantizerEntry<int32_t>(&l2->bias.values, quant_two),
            QuantizerEntry<int8_t>(&pos_eval->weights.values, 1.0),
            QuantizerEntry<int32_t>(&pos_eval->bias.values, quant_two),
        });
    }

    static inline int king_square_index(int relative_king_square) {
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

        return indices[relative_king_square];
    }

    static inline int index(chess::Square piece_square,
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

    void setup_inputs_and_outputs(binpackloader::DataSet& positions) {
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        auto& target = m_loss->target;

#pragma omp parallel for schedule(static, 64) num_threads(16)
        for (int b = 0; b < positions.size(); b++) {
            const auto& entry = positions[b];

            DataLoader::set_features(b, entry, in1, in2, index);

            float p_value  = DataLoader::get_p_value(positions[b]);
            float w_value  = DataLoader::get_w_value(positions[b]);

            float p_target = 1 / (1 + expf(-p_value * sigmoid_scale));
            float w_target = (w_value + 1) / 2.0f;

            float actual_lambda =
                start_lambda + (end_lambda - start_lambda) * (current_epoch / max_epochs);

            target(b) = (actual_lambda * p_target + (1.0f - actual_lambda) * w_target) / 1.0f;
        }
    }

    void setup_inputs_and_outputs_only(binpackloader::DataEntry& entry) {
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        DataLoader::set_features(0, entry, in1, in2, index);
    }
};

}