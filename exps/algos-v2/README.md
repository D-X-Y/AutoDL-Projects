# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size

# Benchmarking 13 NAS Algorithm

The architecture index can be found by use `api.query_index_by_arch(architecture_string)`.

The final discovered architecture ID on CIFAR-10:
```
DARTS (V1):
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|

DARTS (V2):
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
```

The final discovered architecture ID on CIFAR-100:
```
DARTS (V1):
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|nor_conv_1x1~1|none~2|
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|nor_conv_1x1~1|none~2|
|skip_connect~0|+|skip_connect~0|none~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|

DARTS (V2):
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|nor_conv_1x1~1|skip_connect~2|
|skip_connect~0|+|nor_conv_3x3~0|none~1|+|skip_connect~0|none~1|none~2|
|skip_connect~0|+|nor_conv_1x1~0|none~1|+|nor_conv_3x3~0|skip_connect~1|none~2|
```

The final discovered architecture ID on ImageNet-16-120:
```
DARTS (V1):
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|none~1|nor_conv_3x3~2|
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|none~1|nor_conv_3x3~2|
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|none~1|nor_conv_1x1~2|

DARTS (V2):
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|none~1|skip_connect~2|
```
