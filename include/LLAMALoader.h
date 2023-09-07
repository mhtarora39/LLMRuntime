
#include <string>

#ifndef Wdtype
#define Wdtype float 
#endif


struct Config
{
    int dim;         // Transformer size,
    int hidden_size; // For ffn layer.
    int n_layer;     // numer of layers.
    int n_heads;     // numer of heads.
    int n_kv_heads;  // can be less then query heads i.e multi-query heads.
    int vocab_size;  // vocab size
    int seq_len;     // max sequence length
};

struct TransformerWeights
{
    Wdtype *token_embedding_size; // vocab_size, dim
    Wdtype *rms_att_weight;       // layer,dim
    Wdtype *rms_ffn_weight;       // layer,dim
    Wdtype *wq;                   // [layers , n_heads * head_dim size === self.dims]
    Wdtype *wk;                   // [layers , n_heads * head_dim size === self.dims]
    Wdtype *wv;                   // [layers , n_heads * head_dim size === self.dims]
    Wdtype *wo;                   // [layers , n_heads * head_dim size === self.dims]

    // weights for ffn
    Wdtype *w1; // layer , hidden_dim, dim
    Wdtype *w2; // layer , hidden_dim, dim
    Wdtype *w3; // layer , hidden_dim, dim

    // final rms norm
    Wdtype *rms_final_weights; // (dim,)
    // classifier weights for the logits, no  last layer;
    Wdtype *wcls;
    void read_transformer_block(std::string model_path);
};



