# Building a Trading System with AI Assistance — Without Outsourcing Judgment

*Build with AI. Keep the judgment.*

I am not an AI engineer. I am a trader who builds his own tools.

What I am building is a trading system that exports its own market data, and a small local LLM that reads that data and gives me a fast, structured read on the market context after each cycle. Not predictions. A second pair of eyes.

AI is helping me build it faster — exploring ideas, shaping interfaces, writing and refactoring code, iterating through possibilities that would otherwise take weeks. But in trading, moving fast and being right are two very different things.

A system can look complete before it is reliable. A model can sound confident without being useful. A clean piece of AI-generated code can still encode a bad assumption. So this project is really about one question: **where do you let AI in, and where do you keep the decision for yourself?**

## What the model actually returns

Before the philosophy, here is the concrete thing I want out of the model. After each market-processing cycle, it reads a compact snapshot from the engine and returns something like this:

```json
{
  "market_regime": "transition",
  "risk_level": "cautious",
  "key_observations": [
    "Higher-timeframe trend remains positive",
    "Short-term volatility has increased",
    "Breakout volume is weaker than expected"
  ],
  "risk_flags": [
    "Existing long exposure is elevated"
  ],
  "recommended_posture": "avoid adding new exposure"
}
```

This is not a trade instruction. It is a market and risk assessment — a compact explanation of what the system just saw, what looks unusual, and how cautious the current posture should be. Everything below is about how I get to that output, and why I deliberately keep it small.

## The goal is not price prediction

I am not trying to train an AI to predict the next candle.

I do not expect a local LLM to know where price will go. It will not place trades, change position sizes, move stops, or override the execution engine. The trading engine stays responsible for those things, with rules that are explicit, deterministic, and testable.

What I want from the model is narrower, but potentially more useful:

> After each market-processing cycle, give me a fast, structured assessment of the market context and the risks that matter right now.

For medium-frequency trading, the problem is usually not a lack of data. The engine already sees price movement, trend across multiple timeframes, volatility, volume, strategy signals, positions, and exposure.

The difficulty is that these signals arrive as separate pieces. A valid entry signal may appear while volatility is expanding. A higher-timeframe trend may still be intact while short-term structure becomes unstable. Existing exposure may make an otherwise reasonable setup less attractive.

None of that necessarily means "do not trade." But it does mean the system needs a better understanding of the environment it is trading in.

## Enough context, not more data

The answer is not to dump every raw candle and indicator into a model.

The engine does the numerical work first: it calculates indicators, detects market structure, tracks positions, evaluates exposure, and flags anomalies. Then it exports a compact snapshot with only what matters for the current cycle:

- Trend and price structure across relevant timeframes
- Volatility and volume conditions
- Liquidity or abnormal-movement signals
- Strategy state and current positions
- Exposure and risk constraints the engine has already identified

The model does not need to discover these facts from raw data. It needs to interpret their combined meaning.

> Enough context to understand the market. Small enough to stay fast.

This is also why running locally matters. The value is not that the model knows everything about finance — it comes from handing it a well-prepared, domain-specific picture of what the engine just observed.

## Why a small model, running locally

Large models are capable, but raw capability is not the constraint here.

I need an assessment *shortly* after the engine finishes a cycle. If the response comes too late, it is describing a market state that no longer exists. If it depends on an external API, latency and availability become part of the risk. If the output is too broad or verbose, it becomes harder to use in an operational loop.

A small local LLM has a narrower job:

- Understand the trading vocabulary and schemas the system defines
- Read a compact market snapshot
- Return a short, predictable assessment
- Do it fast enough to be useful before the next cycle

The target is not general intelligence. It is fast, consistent interpretation inside a tightly defined domain.

## A deliberately boring architecture

```text
Market data
  → Trading engine calculates signals and risk state
  → Compact market snapshot
  → Local LLM assessment
  → Structured output for dashboard and risk layer
  → Execution stays rule-based inside the engine
```

The model does not return open-ended advice. It returns a structured result the dashboard, operator, or predefined risk rules can act on — like the JSON above.

From there the engine decides what to do using rules that already exist. It might only display the assessment. It might notify me. It might feed selected fields into a predefined risk gate. But execution authority stays outside the model.

## AI can accelerate the build, not own the risk

AI assistance is what made this project feasible to even attempt solo. It is great for the early stages: turning rough ideas into prototypes, generating scaffolding, cleaning up interfaces, clarifying schemas, surfacing implementation options.

But building faster has a hidden cost — it can create the illusion that progress equals correctness. In a trading system, that illusion is expensive.

The core questions cannot be delegated:

- Is this risk rule actually valid?
- Is the input data reliable and current?
- Does this output help, or does it only *sound* helpful?
- What happens when the model is wrong, slow, or unavailable?
- Which decisions must stay deterministic?

Those are design and judgment problems, not prompting problems. AI can help me reason about the market and build the system. But the responsibility for the system's boundaries stays with me.

## What I am trying to build

Not an oracle. Not an autonomous trader.

A local market analyst that sits *beside* a trading engine — one that reads the state the engine has already calculated and turns it into a fast, risk-aware view of the present market.

If it makes exposure easier to understand, highlights unstable conditions, or reduces the chance of overtrading in the wrong environment, then it has earned its place. That is the experiment.

---

I am building this in the open. The code is rough in places and still moving, but if the idea resonates — or if you think I am wrong about where the boundaries should be — I would genuinely like to hear it.

**Repo:** [github.com/hananguyn/candle-training](https://github.com/hananguyn/candle-training)

*Build with AI. Keep the judgment.*
