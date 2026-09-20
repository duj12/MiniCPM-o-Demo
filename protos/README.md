# protos/ —— 从 TTS 仓库同步的 gRPC 生成代码

本目录让本仓库**不再依赖同级的 `TTS` 仓库**就能连 TTS 服务。

| 项 | 值 |
|---|---|
| 源仓库 | `git@git.xmov.ai:DL/TTS.git` |
| 源路径 | `TTS/protos/` |
| 同步时的 HEAD | `d5e65ba`（2025-10-23） |
| protos 最后一次改动 | `bdd6798`（2025-10-09） |
| 同步日期 | 2026-09-20 |

校验完整性：

```bash
sha256sum -c protos/SHA256SUMS
```

## ⚠️ 不要在本仓库重新生成

`tts_pb2.py` 里的 `# source: tts/tts.proto` 会被编进 `AddSerializedFile(...)`，
也就是**注册到 protobuf 全局 descriptor pool 里的类型名**。在别的目录用
`protoc -I protos protos/tts.proto` 生成，得到的是 `# source: tts.proto` ——
生成的代码本身能 import，但**与服务端注册的类型名不一致**，症状是 RPC 报类型
不匹配，而且很难联想到是「生成时的工作目录不对」。

**改 proto 的正确流程**永远是：改 TTS 仓库 → **在 TTS 仓库里**原路径生成 →
原样拷过来。

## 重新生成（在 TTS 仓库里做）

```bash
cd /path/to/TTS
python -m grpc_tools.protoc -I. \
    --python_out=. --grpc_python_out=. --pyi_out=. tts.proto

# 然后**原样**拷贝 —— 不要改 import、不要改 -I、不要改名：
cp protos/{__init__.py,tts.proto,tts_pb2.py,tts_pb2.pyi,tts_pb2_grpc.py} \
   /path/to/MiniCPM-o-Demo/protos/
cd /path/to/MiniCPM-o-Demo/protos && sha256sum __init__.py tts.proto \
   tts_pb2.py tts_pb2.pyi tts_pb2_grpc.py > SHA256SUMS
```

## 为什么不放在 `orchestrator/tts/protos/`

`tts_pb2_grpc.py` 第 5 行是**绝对 import**：

```python
from protos import tts_pb2 as tts_dot_tts__pb2
```

所以包名必须是 `protos`。放在 `orchestrator/tts/` 下面的话，要么改掉
`DO NOT EDIT` 的生成文件（此后每次同步都要重新打补丁，也没法再用 sha256 校验），
要么把 `orchestrator/tts` 塞进 `sys.path` —— 那会让 `sys.path` 上出现**两个
`protos` 包**（TTS 仓库那份也可能还在路径上），谁生效取决于插入顺序。
导入顺序依赖是最难查的一类问题，直接放在仓库根、让 `python -m orchestrator.main`
的 CWD 自然命中，最省事也最稳。

## 用它的地方

- [`orchestrator/tts/client.py`](../orchestrator/tts/client.py) 的 `_proto_modules()`
- [`orchestrator/tests/probe_tts.py`](../orchestrator/tests/probe_tts.py)
