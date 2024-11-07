# DiT - Diffusion Transformers 
A Diffusion Transformer with adaptive layer norm zero (adaLN-Zero) conditioning.
## Block DiT
Shape: output.shape = input.shape
1. AdaLN Modulation -> 6 modul params
2. Attention Layers:
- Shift and scale Norm(x)
- Attention
- Weight x (expanded along the sequence dimension)
- Skip connection
3. MLP Layers:
- Shift, Scale Norm(x)
- MLP
- Weight
- Skip connection
## Final Layer
Apply adaptive layer normalization conditioning to the final output and then project it to the desired output shape.
Shape: (b, sequence_lenght, h_size) -> (b, sequence_lenght, patch_size * patch_size * out_channels)
1. AdaLN Modulation:
Generates two modulation parameters: shift and scale, from the conditioning input *c*
2. Modulated Layer Normalization:
- Norm(x)
- Adjusts the normalized values using shift and scale
```
x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```
3. Linear Projection:
- Project x -> patch_size * patch_size * out_channels

## DiT core
### Input 
Control the model's spatial and embedding dimensions, number of transformer blocks, and model capacity.
```input_size, patch_size, in_channels, hidden_size, depth, num_heads, mlp_ratio```
Parameters for label conditioning and the option to model both mean and variance in the output.
```class_dropout_prob, num_classes, learn_sigma```
### Embedder
```
x_embedder #Converts input images to patch embeddings of dimension hidden_size.
t_embedder #Embeds diffusion timesteps into the same dimension for conditioning.
y_embedder #Embeds class labels into hidden_size with dropout for classifier-free guidance.
pos_embed # A fixed sin-cos position embedding is used to encode spatial positions without learnable parameters.
```
### Blocks
1. DiT Block
2. Final Layers
### Weight Initilization
Uses a custom initialize_weights method:
- Applies Xavier initialization to linear layers.
- Sets pos_embed to fixed sin-cos embeddings.
- Zeroes out adaLN modulation layers to ensure stable training.
- Initializes label embedding and timestep embedding layers with normal distribution.

### Methods
1. Unpatchify:
```
    x: (N, T, patch_size**2 * C)
    imgs: (N, H, W, C)
```
Converts the output back from patch embeddings to the spatial image format (N, out_channels, H, W)
2. Forward Pass:
```
"""
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
```
- Embeds x (input image), t (timestep), and y (class labels).
- Computes c (conditioning) by *summing the timestep and label embeddings*.
- Applies each DiT block with adaLN conditioning on c.
- The final layer projects and unpatchifies the output.
3. Classifier-Free Guidance 
- Runs an unconditional and conditional forward pass to compute guided outputs for diffusion sampling.
- Splits the output to apply guidance only to specified channels (e.g., the first three channels).
        
### Output
Without Guidance:
The model produces an output tensor in the spatial format ```(N, out_channels, H, W)```
With Guidance: 
Applies classifier-free guidance to generate an enhanced output by interpolating between conditional and unconditional outputs

## PatchEmbedder
From timm.models.vision_transformer import PatchEmbedder
https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L26
## TimestepEmbedder
Embeds scalar diffusion timesteps into vector representations
1. MLP
Linear(frequency_embedding_size -> hidden_size) -> SiLU -> Linear(hidden_size -> hidden_size)
2. Sinusoidal Embedding
The frequency is controlled by the max_period parameter, and it uses half the embedding size for cos values and the other half for sin values, resulting in an output shape of (N, frequency_embedding_size)
3. Forward Pass
- t is passed to timestep_embedding to create sinusoidal embeddings.
- The sinusoidal embeddings are processed by the MLP to produce the final timestep embedding of shape ```(N, hidden_size)```
## LabelEmbedder
Embeds class labels into vector representations, with an option to drop labels for classifier-free guidance
1. Embedding Table:
An ```nn.Embedding``` layer initialized with ```num_classes + 1``` embeddings if label dropout is used, so that an additional embedding can represent dropped labels.
2. Dropout Probability hyperparameter:
The probability of dropping a label, which is used for classifier-free guidance by encouraging the model to ignore class information at times during training.
3. Token Drop Method:
Drops labels with probability ```dropout_prob```, setting them to a special token that represents an "empty" or "unconditional" label. 
4. Forward Pass:
- If in training mode and ```dropout_prob > 0```, or if specific drop IDs are forced, labels are dropped using token_drop.
- The final label embedding is retrieved from embedding_table based on the modified (or original) labels and returned with shape ```(N, hidden_size)```