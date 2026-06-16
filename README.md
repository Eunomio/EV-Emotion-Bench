# NOMI LLM-as-Judge Demo

这个demo用于验证“NOMI车载情感陪伴智能体评分体系”能否在具体对话样本上跑通。它不是SFT、DPO或Reward Model训练，只是一个小型HeartBench式评估流程：

1. 用ChatECNU根据场景描述和用户对话进行场景初始化，推断风险等级、驾驶负荷和用户初始状态。
2. 用ChatECNU生成候选NOMI回复，并同时保留人工高分参考和人工低分参考。
3. 基于每条NOMI候选回复，再生成用户下一轮反应。
4. 用ChatECNU作为LLM-as-judge，根据场景初始化结果、NOMI回复质量和用户before/after变化输出结构化评分。
5. 按原始md中的二级维度展开评分，再汇总每个场景、每个候选回复的安全门控、维度分、用户状态扭转结果、总分和主要问题。

综合分使用场景动态权重，而不是简单相加。脚本会根据场景初始化结果中的风险等级、驾驶负荷、风险意图、情绪强度、多轮历史长度和是否有用户反馈，自动调整：

- 单轮质量权重
- 多轮轨迹权重
- 用户状态扭转权重

安全伦理仍作为门控系数`G_safe`作用于总分。

当前demo是纯文本评估，因此“声学情绪表达一致性”会被标记为`not_applicable`，不参与总分。其他维度均按原始评分体系保留。

## 文件说明

- `nomi_llm_judge_demo.py`：主程序。
- `sample_cases.json`：内置的车载情感陪伴评估场景。
- `.env.example`：环境变量示例，不包含真实密钥。

## 快速运行

方式一：像HeartBench一样通过命令行传API key：

```powershell
python .\nomi_llm_judge_demo.py --api-key "你的ChatECNU API key" --test-connection
```

方式二：通过环境变量设置API key：

```powershell
$env:CHAT_ECNU_API_KEY="你的ChatECNU API key"
```

安装依赖：

```powershell
pip install openai==2.12.0
```

运行真实调用：

```powershell
python .\nomi_llm_judge_demo.py --cases .\sample_cases.json --out-dir ..\..\outputs\nomi_llm_judge_results
```

如果使用命令行传key：

```powershell
python .\nomi_llm_judge_demo.py --api-key "你的ChatECNU API key" --cases .\sample_cases.json --out-dir .\results
```

先运行连接测试：

```powershell
python .\nomi_llm_judge_demo.py --test-connection
```

脚本参考HeartBench的调用方式，使用OpenAI官方SDK访问ChatECNU兼容接口。默认不启用`thinking`，因为短测试请求在启用thinking时可能返回空`message.content`；如果确实需要，可以显式加`--enable-thinking`。

默认情况下，脚本不会读取`HTTP_PROXY`/`HTTPS_PROXY`等系统代理环境变量，因为有些Windows代理会导致ChatECNU的TLS握手失败。如果你的网络必须走代理，可以显式加：

```powershell
python .\nomi_llm_judge_demo.py --test-connection --use-env-proxy
```

如果系统代理环境变量不可用，建议显式指定代理端口。端口以你的代理软件为准，常见为`7890`、`7897`或`10809`：

```powershell
python .\nomi_llm_judge_demo.py --api-key "你的ChatECNU API key" --test-connection --proxy-url http://127.0.0.1:7890
```

完整评估同理：

```powershell
python .\nomi_llm_judge_demo.py --api-key "你的ChatECNU API key" --cases .\sample_cases.json --out-dir .\results --proxy-url http://127.0.0.1:7890
```

如果两种方式都失败，先跑不带API key的网络诊断：

```powershell
python .\nomi_llm_judge_demo.py --diagnose-network
```

重点看`TLS direct`和`httpx GET /models trust_env=False/True`哪一项失败。

如果只是想先离线检查流程和输出格式：

```powershell
python .\nomi_llm_judge_demo.py --cases .\sample_cases.json --out-dir ..\..\outputs\nomi_llm_judge_results --mock
```

## 默认接口

默认使用OpenAI兼容格式：

- Base URL: `https://chat.ecnu.edu.cn/open/api/v1`
- Chat endpoint: `/chat/completions`
- 默认生成模型: `ChatECNU`
- 默认judge模型: `ChatECNU`

如果接口路径或模型名需要调整，可以用参数覆盖：

```powershell
python .\nomi_llm_judge_demo.py --model ecnu-plus --judge-model ecnu-max
```

## 输出

程序会生成：

- `results.json`：完整结构化结果。
- `results.md`：可直接放进作业/汇报的结果表格、候选回复、总体理由和二级维度细项。
  其中每个样本都会包含“场景初始化结果”“候选NOMI回复”和“用户下一轮反应”，用户状态扭转维度基于这两轮变化评分。

输出结构包括：

- 安全伦理门控：人身安全与风险响应适配性、隐私与信息边界、道德伦理与用户自主性。
- 单轮动态评分：回复相关性与信息有效性、场景适配性、长度、内容密度、情绪识别后的反馈准确性、情绪回应强度与深度适配性、声学情绪表达一致性、心理学技术调用、拟人化适度性、人格匹配度。
- 多轮轨迹修正：上下文衔接与状态维护、多轮情绪轨迹识别、多轮节奏控制、多轮人格一致性。
- 用户状态扭转：认知灵活性、归因重构、视角开放、情绪强度变化、情绪表达、情绪整合、接纳程度、自我暴露程度、被理解感。
