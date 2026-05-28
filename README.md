本项目大部分代码均来自于[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), 本项目仅作了`cli`化与角色特化打包

本项目遵循 MIT 协议

对于模型, 请按照原始项目中的教程下载对应模型并按位置放入[models](./models)目录下, 其中 GPT v3 底模(s1v3.ckpt)应该直接放在[models/](./models/)下

对于参考音频, 请按照台词命名, 并将主参考音频放在[refs/main](./refs/main)下, 辅助参考音频放在[refs/sub](./ref/sub)下, 此处预提供了几段来自`明日方舟`官方动画的语音切片, 仅供交流学习使用, 版权归原公司所有

对于其它音色模型, 必须是由原始项目训练得到的pth权重, 取名为`prts.pth`后放入[models/SoVITS](./models/SoVITS), 当然你也可以自行修改相关逻辑

普瑞赛斯声线模型请自行训练, 或者使用 [B站视频](https://b23.tv/W1E3hML) 大佬开源的已训练好的模型

环境: Python 3.10 (必须低于3.13, 因为记得有个必须库在3.13被废弃了会很麻烦)

依赖: 请查看原项目, 此处不赘述

---

参数说明

```plaintext
[Required Arguments]

w/word <str> 也就是目标合成文本

---

[Optional Arguments]                | [Default]        | [Choices]

-s/--sampling                 <int> | 64               | { x | x ∈  4, 8, 16, 32, 64, 128 }
-o/--output-path              <str> | workspace/output/{yy-MM-dd}/{word}.wav
-tk/--topk                    <int> | 5                | { x | x >= 1, x <= 100 }
-tp/--topp                    <int> | 1                | { x | x >= 0, x <= 1 }
-t/--temperature              <int> | 1                | { x | x >= 0, x <= 1 }
-b/--batch-size               <int> | 20               | { x | x >= 1, x <= 200 }
-p/--paragraph-separation   <float> | 0.3              | { x | x >= 0.01, x <= 1}
-rp/--repetition-punishment <float> | 1.35             | { x | x >= 0, x <= 2 }
-r/--speech-rate            <float> | 1                | { x | x >= 0.6, x <= 1.65}
-l/--language                 <str> | JP               | EN, CN, JP, KOR
-r/--random-seed              <int> | Random
```
