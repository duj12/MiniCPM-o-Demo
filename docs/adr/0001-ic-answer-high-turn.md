# IC ANSWER 必须以 HIGH 话轮为准

IC 判 ANSWER 时，话轮完成度必须是 HIGH。StartLux 的 `repair=answer` / `repair=burst` 只是“听感可用”的意见：它可以挡住“请再说一遍”这类听感 UTTER，但不能触发 ANSWER。一个已经到达 HIGH 的话轮可以先用「待回答话轮」冻结转写和回答证据，跨过等待 StartLux 结束判断的那段时间（上限 1.5 秒），判断回来或等待超时后再回答或说结束语。

## Considered Options

- **接受 `repair=answer` 直接 ANSWER（改动前的行为）**：回放 `s-9554c311559e-raw.mkv` 时，LOW 的半句被提前回答，同一句到 HIGH 后又被答一遍，随后新起的 TTS 在 0.00s 被 YIELD 掐断。LOW 的定义是“肯定没说完”，用它触发 ANSWER 本身就是错的。
- **把 ANSWER 放宽到 MEDIUM**：生产里的语音感知只有 HIGH 和 LOW，没有 MEDIUM。为一个不存在的档位修改契约无法落地，也会把“肯定没说完”重新变成可回答。
- **一律等 StartLux 返回结束判断再回答**：StartLux 慢或挂掉时会丢掉已经说完整的 HIGH 话轮，用户说完却得不到回答。待回答话轮冻结证据、1.5 秒超时兜底，既守住 HIGH 闸，又不丢已经到过 HIGH 的话轮。
- **给 `repair=burst` 开非 HIGH 回答的口子**：`burst` 本身不要求 HIGH；让它绕过 HIGH 闸等于重新引入半句回答。

## Consequences

- LOW 话轮只等待，不回答；`repair=answer` / `repair=burst` 在 LOW 时只影响听感 UTTER，不产生 ANSWER。
- 回答所需的转写、身份、主体、情感、语言、噪声和 SNR 由 ANSWER 动作自己携带，runtime 不再从已经被清空的 ASR 状态读取。
- HIGH 话轮不会因为 StartLux 结束判断晚到而丢失。
