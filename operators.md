# Currently supported operations

| Operation type            | Support            | Opset versions     |
|---------------------------|--------------------|--------------------|
| Abs                       | Supported          | [6, 13]            |
| Acos                      | Supported          | [7]                |
| Acosh                     | Can't be supported | []                 |
| Add                       | Supported          | [14, 13, 7, 6, 1]  |
| And                       | Supported          | [7, 1]             |
| ArgMax                    | Planned            | []                 |
| ArgMin                    | Planned            | []                 |
| Asin                      | Supported          | [7]                |
| Asinh                     | Can't be supported | []                 |
| Atan                      | Supported          | [7]                |
| Atanh                     | Can't be supported | []                 |
| AveragePool               | Supported          | [11, 10, 7]        |
| BatchNormalization        | Supported          | [9, 14, 15]        |
| BitShift                  | Not supported      | []                 |
| Cast                      | Supported          | [13, 9]            |
| Ceil                      | Supported          | [6, 13]            |
| Clip                      | Supported          | [13, 12, 11, 6]    |
| Compress                  | Not supported      | []                 |
| Concat                    | Supported          | [13, 11, 4]        |
| ConcatFromSequence        | Not supported      | []                 |
| Constant                  | Supported          | [13, 12, 11, 9]    |
| ConstantOfShape           | Supported          | [9]                |
| Conv                      | Supported          | [11, 1]            |
| ConvInteger               | Not supported      | []                 |
| ConvTranspose             | Supported          | [11, 1]            |
| Cos                       | Supported          | [7]                |
| Cosh                      | Can't be supported | []                 |
| CumSum                    | Planned            | []                 |
| DepthToSpace              | Not supported      | []                 |
| DequantizeLinear          | Not supported      | []                 |
| Det                       | Not supported      | []                 |
| Div                       | Supported          | [14, 13, 7, 6, 1]  |
| Dropout                   | Planned            | []                 |
| Einsum                    | Not supported      | []                 |
| Elu                       | Supported          | [6]                |
| Equal                     | Supported          | [13, 11, 7]        |
| Erf                       | Supported          | [13, 9]            |
| Exp                       | Supported          | [13, 6]            |
| Expand                    | Supported          | [13, 8]            |
| EyeLike                   | Planned            | []                 |
| Flatten                   | Supported          | [9, 11, 13]        |
| Floor                     | Supported          | [6, 13]            |
| GRU                       | Not supported      | []                 |
| Gather                    | Supported          | [13, 11, 1]        |
| GatherElements            | Supported          | [13, 11]           |
| GatherND                  | Planned            | []                 |
| Gemm                      | Supported          | [13, 11, 9]        |
| GlobalAveragePool         | Supported          | [1]                |
| GlobalLpPool              | Not supported      | []                 |
| GlobalMaxPool             | Not supported      | []                 |
| Greater                   | Supported          | [13, 9, 7]         |
| GridSample                | Not supported      | []                 |
| HardSigmoid               | Supported          | [6, 1]             |
| Hardmax                   | Planned            | []                 |
| Identity                  | Supported          | [1, 13, 14, 16]    |
| If                        | Not supported      | []                 |
| InstanceNormalization     | Planned            | []                 |
| IsInf                     | Not supported      | []                 |
| IsNaN                     | Not supported      | []                 |
| LRN                       | Not supported      | []                 |
| LSTM                      | Not supported      | []                 |
| LeakyRelu                 | Supported          | [6, 1]             |
| Less                      | Supported          | [13, 9, 7]         |
| Log                       | Supported          | [6, 13]            |
| Loop                      | Not supported      | []                 |
| LpNormalization           | Planned            | []                 |
| LpPool                    | Planned            | []                 |
| MatMul                    | Supported          | [13, 9, 1]         |
| MatMulInteger             | Not supported      | []                 |
| Max                       | Planned            | []                 |
| MaxPool                   | Supported          | [8, 10, 11, 12]    |
| MaxRoiPool                | Not supported      | []                 |
| MaxUnpool                 | Not supported      | []                 |
| Mean                      | Planned            | []                 |
| Min                       | Planned            | []                 |
| Mod                       | Planned            | []                 |
| Mul                       | Supported          | [14, 13, 7, 6, 1]  |
| Multinomial               | Not supported      | []                 |
| Neg                       | Planned            | []                 |
| NonMaxSuppression         | Supported          | [11, 10]           |
| NonZero                   | Planned            | []                 |
| Not                       | Supported          | [1]                |
| OneHot                    | Not supported      | []                 |
| Optional                  | Not supported      | []                 |
| OptionalGetElement        | Not supported      | []                 |
| OptionalHasElement        | Not supported      | []                 |
| Or                        | Supported          | [7, 1]             |
| PRelu                     | Planned            | []                 |
| Pad                       | Supported          | [13, 11, 2]        |
| Pow                       | Supported          | [15, 13, 12, 7, 1] |
| QLinearConv               | Not supported      | []                 |
| QLinearMatMul             | Not supported      | []                 |
| QuantizeLinear            | Not supported      | []                 |
| RNN                       | Not supported      | []                 |
| RandomNormal              | Planned            | []                 |
| RandomNormalLike          | Planned            | []                 |
| RandomUniform             | Planned            | []                 |
| RandomUniformLike         | Planned            | []                 |
| Reciprocal                | Not supported      | []                 |
| ReduceL1                  | Supported          | [13, 11, 1]        |
| ReduceL2                  | Supported          | [13, 11, 1]        |
| ReduceLogSum              | Supported          | [13, 11, 1]        |
| ReduceLogSumExp           | Supported          | [13, 11, 1]        |
| ReduceMax                 | Supported          | [13, 12, 11, 1]    |
| ReduceMean                | Supported          | [13, 11, 1]        |
| ReduceMin                 | Supported          | [13, 12, 11, 1]    |
| ReduceProd                | Supported          | [13, 11, 1]        |
| ReduceSum                 | Supported          | [11, 1, 13]        |
| ReduceSumSquare           | Supported          | [13, 11, 1]        |
| Relu                      | Supported          | [14, 13, 6]        |
| Reshape                   | Supported          | [14, 13, 5]        |
| Resize                    | Supported          | [10, 13, 11]       |
| ReverseSequence           | Not supported      | []                 |
| RoiAlign                  | Supported          | [10]               |
| Round                     | Supported          | [11]               |
| Scan                      | Not supported      | []                 |
| Scatter(deprecated)       | Not supported      | []                 |
| ScatterElements           | Not supported      | []                 |
| ScatterND                 | Supported          | [13, 11]           |
| Selu                      | Supported          | [6]                |
| SequenceAt                | Not supported      | []                 |
| SequenceConstruct         | Not supported      | []                 |
| SequenceEmpty             | Not supported      | []                 |
| SequenceErase             | Not supported      | []                 |
| SequenceInsert            | Not supported      | []                 |
| SequenceLength            | Not supported      | []                 |
| Shape                     | Supported          | [15, 13, 1]        |
| Shrink                    | Planned            | []                 |
| Sigmoid                   | Supported          | [13, 6, 1]         |
| Sign                      | Supported          | [9, 13]            |
| Sin                       | Supported          | [7]                |
| Sinh                      | can't support      | []                 |
| Size                      | Planned            | []                 |
| Slice                     | Supported          | [9, 13, 11, 10]    |
| Softplus                  | Supported          | [1]                |
| Softsign                  | Supported          | [1]                |
| SpaceToDepth              | Planned            | []                 |
| Split                     | Supported          | [13, 2, 11]        |
| SplitToSequence           | Not supported      | []                 |
| Sqrt                      | Supported          | [13, 6, 1]         |
| Squeeze                   | Supported          | [11, 1, 13]        |
| StringNormalizer          | Not supported      | []                 |
| Sub                       | Supported          | [14, 13, 7, 6, 1]  |
| Sum                       | Planned            | []                 |
| Tan                       | Supported          | [7]                |
| Tanh                      | Supported          | [6, 13]            |
| TfIdfVectorizer           | Not supported      | []                 |
| ThresholdedRelu           | Not supported      | []                 |
| Tile                      | Supported          | [13, 6]            |
| TopK                      | Supported          | [11, 10, 1]        |
| Transpose                 | Supported          | [13, 1]            |
| Trilu                     | Planned            | []                 |
| Unique                    | Planned            | []                 |
| Unsqueeze                 | Supported          | [11, 1, 13]        |
| Upsample(deprecated)      | Planned            | []                 |
| Where                     | Supported          | [16, 9]            |
| Xor                       | Supported          | [7, 1]             |
| Function                  | Not supported      | []                 |
| Bernoulli                 | Planned            | []                 |
| CastLike                  | Planned            | []                 |
| Celu                      | Supported          | [12]               |
| DynamicQuantizeLinear     | Not supported      | []                 |
| GreaterOrEqual            | Supported          | [12]               |
| HardSwish                 | Supported          | [14]               |
| LessOrEqual               | Supported          | [12]               |
| LogSoftmax                | Supported          | [13, 11, 1]        |
| MeanVarianceNormalization | Planned            | []                 |
| NegativeLogLikelihoodLoss | Planned            | []                 |
| Range                     | Supported          | [11]               |
| SequenceMap               | Not supported      | []                 |
| Softmax                   | Supported          | [11, 1, 13]        |
