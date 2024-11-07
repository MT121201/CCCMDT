# DiT - Diffusion Transformers
A Diffusion Transformer with adaptive layer norm zero (adaLN-Zero) conditioning.

## Block DiT
Shape: `output.shape = input.shape`

1. AdaLN Modulation → 6 modulation parameters
2. Attention Layers:
   - Shift and scale Norm(x)
   - Attention
   - Weighted by `gate_msa` (expanded along the sequence dimension)
   - Skip connection
3. MLP Layers:
   - Shift, scale Norm(x)
   - MLP
   - Weighted by `gate_mlp`
   - Skip connection

## Final Layer
Applies adaptive layer normalization conditioning to the final output and projects to desired output shape.  
Shape: `(b, sequence_length, h_size) -> (b, sequence_length, patch_size * patch_size * out_channels)`

1. **AdaLN Modulation**: Generates shift and scale modulation parameters from conditioning input *c*
2. **Modulated Layer Normalization**:
   - Norm(x)
   - Adjusts normalized values with shift and scale  
```x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)```
3. **Linear Projection**: Projects `x` to `patch_size * patch_size * out_channels`

## DiT Core
### Input
Controls model's spatial and embedding dimensions, number of transformer blocks, and model capacity.  
Parameters: `input_size, patch_size, in_channels, hidden_size, depth, num_heads, mlp_ratio`  
Label conditioning and output modeling options: `class_dropout_prob, num_classes, learn_sigma`

### Embedders
```
x_embedder # Converts input images to patch embeddings of dimension hidden_size. 
t_embedder # Embeds diffusion timesteps into hidden_size for conditioning. 
y_embedder # Embeds class labels into hidden_size with dropout for classifier-free guidance. 
pos_embed # Fixed sin-cos position embedding for spatial encoding without learnable parameters.
```

### Blocks
1. **DiT Block**: Processes sequences with adaLN conditioning
2. **Final Layer**: Modulates and projects to output shape

### Weight Initialization
- **Custom `initialize_weights` method**:
  - Xavier initialization for linear layers.
  - Fixed sin-cos embeddings for `pos_embed`.
  - Zeroed adaLN modulation layers for stability.
  - Normal distribution for label and timestep embedding layers.

### Methods
1. **Unpatchify**  
```x: (N, T, patch_size**2 * C) imgs: (N, H, W, C)```
Converts output back to spatial image format `(N, out_channels, H, W)`

2. **Forward Pass**
Forward pass of DiT. 
```
x: (N, C, H, W) #spatial input tensor (images or latent representations) 
t: (N,) #diffusion timesteps 
y: (N,) #class labels
```
- Embeds `x` (input image), `t` (timestep), and `y` (class labels).
- Computes conditioning `c` by summing timestep and label embeddings.
- Applies each DiT block with adaLN conditioning on `c`.
- Final layer projects and unpatchifies the output.

3. **Classifier-Free Guidance**  
- Runs unconditional and conditional forward passes for guided diffusion sampling.
- Splits output to apply guidance only to specified channels (e.g., first three channels).

### Output
- **Without Guidance**: Produces tensor in spatial format `(N, out_channels, H, W)`  
- **With Guidance**: Interpolates conditional and unconditional outputs for enhanced results

## PatchEmbedder
`From timm.models.vision_transformer import PatchEmbedder`  
[Link to source code](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L26)

## TimestepEmbedder
Embeds scalar diffusion timesteps into vector representations.

1. **MLP**  
`Linear(frequency_embedding_size -> hidden_size) -> SiLU -> Linear(hidden_size -> hidden_size)`

2. **Sinusoidal Embedding**  
- Controlled by `max_period`, uses `cos` and `sin` values over half the embedding size, resulting in shape `(N, frequency_embedding_size)`

3. **Forward Pass**  
- `t` passed to `timestep_embedding` for sinusoidal embeddings.
- MLP processes embeddings to final shape `(N, hidden_size)`

## LabelEmbedder
Embeds class labels with optional dropout for classifier-free guidance.

1. **Embedding Table**  
`nn.Embedding` layer with `num_classes + 1` embeddings if dropout is used, where additional embedding represents dropped labels.

2. **Dropout Probability**  
- Probability of dropping a label for classifier-free guidance during training.

3. **Token Drop Method**  
- Drops labels with probability `dropout_prob`, setting to special token for "empty" or "unconditional" label.

4. **Forward Pass**  
- If training and `dropout_prob > 0`, or if forced, uses `token_drop`.
- Retrieves final label embedding from `embedding_table` with shape `(N, hidden_size)`
