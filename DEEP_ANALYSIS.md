# Deep Analysis of Ludic

## 1. Executive Summary

**Ludic** is a research-grade LLM-RL (Large Language Model Reinforcement Learning) library designed for **agentic behavior** - multi-step reasoning with tool use, not single-step LLM completions. It's built with the philosophy of being a **library, not a framework** - loosely coupled components you can swap and extend freely.

The name appears to be a play on "ludic" (relating to play/games) - fitting for an RL library where agents learn through interaction.

---

## 2. Architectural Philosophy

### 2.1 Core Design Principles

The codebase follows several key architectural decisions that distinguish it from other LLM-RL frameworks:

**1. Clean Agent/Environment Separation**
```
Agent ≠ Environment
```
- **Environments** are pure state-transition functions emitting rewards
- **Agents** are LLMs with state, context management, parsing, and tools
- **Interaction Protocols** explicitly define the agent-env loop

**2. Token-In Inference (Drift-Free Training)**
- Agents apply chat templates locally and send pre-tokenized prompts
- Uses vLLM's `/v1/completions` endpoint (not `/v1/chat/completions`)
- Ensures exact alignment between training and inference tokens

**3. RL Algorithm = Credit Assigner + Loss**
- Decouples *what credit each step receives* from *how to compute the loss*
- One trainer, many algorithms via composition

**4. AgentStep vs EnvironmentStep Distinction**
- Training needs the **full reasoning trace**, not just final actions
- A ReAct agent calling tools 3 times produces 3 AgentSteps with token traces
- EnvironmentStep captures state transitions from `env.step()`

---

## 3. Core Data Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                          ROLLOUT GENERATION                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  RolloutRequest ──► RolloutEngine ──► InteractionProtocol.run()        │
│                           │                    │                       │
│                           │                    ▼                       │
│                           │           ┌───────────────┐                │
│                           │           │    Agent      │                │
│                           │           │  • ChatClient │                │
│                           │           │  • Context    │                │
│                           │           │  • Parser     │                │
│                           │           │  • ChatTemplate                │
│                           │           └───────┬───────┘                │
│                           │                   │                        │
│                           ▼                   ▼                        │
│                    ┌────────────┐      ┌────────────┐                  │
│                    │ Environment│◄────►│  Protocol  │                  │
│                    │  .step()   │      │  Loop      │                  │
│                    └────────────┘      └────────────┘                  │
│                           │                   │                        │
│                           └───────┬───────────┘                        │
│                                   ▼                                    │
│                             Rollout[]                                  │
│                      (AgentSteps + EnvSteps)                           │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                            TRAINING                                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Rollout[] ──► CreditAssigner.compute() ──► SAWItems                   │
│                       │                         │                      │
│                       │   (weight per step)     │                      │
│                       ▼                         ▼                      │
│              ┌─────────────────┐      ┌─────────────────┐              │
│              │ SAWBatch        │      │ Micro-batching  │              │
│              │ • input_ids     │◄─────│ • token budget  │              │
│              │ • action_mask   │      │ • collation     │              │
│              │ • weight        │      └─────────────────┘              │
│              │ • actor_logps   │                                       │
│              └────────┬────────┘                                       │
│                       │                                                │
│                       ▼                                                │
│              ┌─────────────────┐                                       │
│              │ RLAlgorithm     │                                       │
│              │ .compute_loss() │                                       │
│              └────────┬────────┘                                       │
│                       │                                                │
│                       ▼                                                │
│              loss.backward() ──► optimizer.step() ──► sync_weights()   │
│                                                              │         │
│                                                              ▼         │
│                                                         vLLM server    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Types and Data Structures

### 4.1 Token Trace (`types.py:19`)

```python
@dataclass(frozen=True)
class TokenTrace:
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    completion_logprobs: Optional[List[float]] = None
    finish_reason: Optional[str] = None
```

The canonical representation of a model call - **single source of truth** for training.

### 4.2 Step Types

**AgentStep** (`types.py:167`) - Every model call:

```python
@dataclass
class AgentStep:
    index: int
    prompt_messages: List[Message]
    action: str
    action_target: str  # "internal" | "external" | "env"
    trace: TokenTrace   # For training
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[List[Dict]]
    ...
```

**EnvironmentStep** (`types.py:188`) - State transitions:

```python
@dataclass
class EnvironmentStep:
    prev_obs: Observation
    parsed_action: Any
    next_obs: Optional[Observation]
    source_agent_step_id: str  # Links back to agent
    agent_step_ids: List[str]  # All agent steps in this turn
    ...
```

### 4.3 SAWItem (State-Action-Weight)

The fundamental training unit (`training/types.py`):

```python
@dataclass
class SAWItem:
    input_ids: List[int]      # Full sequence (prompt + completion)
    attention_mask: List[int]
    action_mask: List[int]    # 1 for completion tokens, 0 for prompt
    weight: float             # Credit from credit assigner
    actor_logps: Optional[ActorTokenLogps]  # For ratio-based methods
    meta: Dict[str, Any]
```

---

## 5. The RL Algorithm System

### 5.1 Credit Assignment

Maps `List[Rollout] → Dict[(rollout_id, step_index), weight]`

| Assigner | Formula | Use Case |
|----------|---------|----------|
| **MonteCarloReturn** | `G_t = r_t + γ·G_{t+1}` | Classic REINFORCE |
| **GroupNormalizedReturn** | `A_i = R_i - mean(R_group)` | GRPO-style |
| **HybridNormalizedReturn** | Group-mean + batch-std | ScaleRL |
| **ConstantCredit** | `w = 1.0` | SFT/behavioral cloning |

### 5.2 Loss Functions

Consume `(logits, batch) → (loss, stats)` with memory-efficient `SharedContext`:

| Loss | Key Formula | Notable Feature |
|------|-------------|-----------------|
| **ReinforceLoss** | `-E[sg(r) · A · log π]` | IS-corrected REINFORCE |
| **TokenClippedSurrogateLoss** | Token-level PPO clipping | GRPO default |
| **ClippedSurrogateLoss** | Sequence-level clipping | GSPO-style |
| **CISPOLoss** | Clip IS-weight, not update | Preserves reflective tokens |
| **SAPOLoss** | Soft sigmoid gate | Smooth trust region |
| **GMPOLoss** | Geometric mean of ratios | Robust to outliers |

### 5.3 Algorithm Presets (`algorithm.py`)

Factory functions combine credit + loss:

```python
# GRPO: Group-relative with token-level clipping
make_grpo(group_size=4)

# GSPO: Tighter sequence-level clipping
make_gspo(group_size=4, clip_eps_low=3e-4, clip_eps_high=4e-4)

# CISPO: Preserves rare reasoning tokens
make_cispo(group_size=4)

# SAPO: Soft adaptive gating
make_sapo(group_size=4, tau_pos=1.0, tau_neg=1.05)

# GMPO: Geometric mean ratios
make_gmpo(group_size=4, clip_eps_low=0.4, clip_eps_high=0.4)

# ScaleRL: Production recipe
make_scalerl(group_size=4)

# SFT: Behavioral cloning
make_sft()
```

---

## 6. The Agent System

### 6.1 Agent Hierarchy

```
Agent (base_agent.py)
  └── Stateful LLM harness
      • ChatClient (inference backend)
      • ContextStrategy (memory management)
      • Parser (action extraction)
      • ChatTemplate (tokenization)

ToolAgent (tool_agent.py)
  └── Adds tool calling
      • tools: Internal (agent executes)
      • external_tools: Protocol handles

ReActAgent (react_agent.py)
  └── Think → Tool* → Act loop
      • action_target: internal/external/env
```

### 6.2 Context Strategies

Abstract memory management with lifecycle hooks:

```python
class ContextStrategy:
    def on_env_reset(self, obs, info): ...
    def on_before_act(self) -> List[Message]: ...
    def on_after_act(self, response): ...
    def on_after_step(self, obs, info): ...
```

Implementations:
- `FullDialog`: Keep entire history
- `TruncatedThinkingContext`: Truncate `<think>` blocks in prompts

---

## 7. The Interaction Protocol System

Defines **how** agents and environments interact:

```python
class InteractionProtocol:
    async def run(
        self,
        env: LudicEnv,
        max_steps: int,
        ...
    ) -> List[Rollout]
```

### 7.1 SingleAgentProtocol (`single_agent.py`)

Standard single-agent loop with:
- Parser failure handling (synthetic observations)
- External tool handler for delegation
- Time-limit truncation tracking

```
while not (terminated or truncated):
    act_result = agent.act()

    for step in act_result.steps:
        if action_target == "external":
            result = external_tool_handler(tool_calls)
            feed_back_to_agent()
            continue

        if action_target == "env":
            if parse_error:
                feed_synthetic_obs()
            else:
                outcome = env.step(parsed_action)
                record_env_step()
```

### 7.2 action_target Semantics

| Target | parse_result | Behavior |
|--------|--------------|----------|
| `"internal"` | set | Agent handled internally, loop continues |
| `"external"` | None | Protocol calls handler, feeds result back |
| `"env"` | set | Parse → env.step() → EnvironmentStep |

---

## 8. The Trainer

### 8.1 Training Loop (`trainer.py`)

```python
async def train_step(self):
    # 1. Fetch macro-batch from batch source
    saw_batch = await batch_source.next_batch()

    # 2. Preprocess (validate actor_logps, filter zeros)
    if algo.preprocess:
        saw_batch = algo.preprocess(saw_batch)

    # 3. Split into micro-batches by token budget
    micro_chunks = split_items_by_token_budget(...)

    # 4. Accumulate gradients
    for chunk in micro_chunks:
        batch = collate_saw_items(chunk)
        loss, stats = algo.compute_loss(model, batch)
        scaled_loss = loss * (len(chunk) / total)
        scaled_loss.backward()

    # 5. Optimizer step
    clip_grad_norm_()
    optimizer.step()
    optimizer.zero_grad()

    # 6. Sync weights to vLLM
    if should_sync:
        publisher.publish(state_dict)
```

### 8.2 FSDP2 Awareness

The trainer is **FSDP2-aware**:
- Disables gradient sync on non-final micro-batches
- Gathers full state dict via DCP APIs for checkpointing
- Handles LoRA merge/unmerge for weight pushing

---

## 9. The Batching System

### 9.1 Batch Sources

```python
class BatchSource(Protocol):
    async def next_batch(self) -> SAWBatch: ...
```

| Source | Description |
|--------|-------------|
| `RolloutBatchSource` | Synchronous - blocks on rollout generation |
| `OfflineBatchSource` | Loads pre-collected JSONL rollouts |
| `PipelineBatchSource` | Actor/learner split via Redis |

### 9.2 Request Expansion (Intra-Batch Control)

For GRPO-style algorithms, expand each request into G variants:

```python
class GRPORequestStrategy:
    def expand(self, request) -> List[RolloutRequest]:
        # Same env seed, different sampling seeds
        return [request.with_group_id(i) for i in range(group_size)]
```

---

## 10. Memory Efficiency

### 10.1 SharedContext for Composite Losses

When combining multiple losses (e.g., policy + KL), the expensive `log_softmax` over `[B, T, V]` logits is computed **once** and cached:

```python
class SharedContext:
    @property
    def token_logp(self) -> Tensor:
        if "token_logp" not in self._cache:
            self._cache["token_logp"] = selective_log_softmax(...)
        return self._cache["token_logp"]
```

Memory savings (7B model, V=32K, B=8, T=4096):
- Without sharing: 2 losses × [B,T,V] ≈ **4GB**
- With sharing: 1 × [B,T,V] ≈ **2GB**

### 10.2 Selective Log Softmax

```python
@torch.compile(dynamic=True)
def selective_log_softmax(logits, index):
    # Inductor fuses into single kernel
    # Avoids materializing full [B, T, V] probability tensor
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index.unsqueeze(-1)).squeeze(-1)
```

---

## 11. Environments

### 11.1 LudicEnv Interface (`envs/env.py`)

Multi-agent by default:

```python
class LudicEnv(ABC, Generic[AgentID, ObsType, ActionType]):
    @property
    def agent_ids(self) -> List[AgentID]: ...

    @property
    def active_agents(self) -> List[AgentID]: ...

    def reset(self, seed) -> Dict[AgentID, Tuple[Obs, Info]]: ...

    def step(self, actions) -> Dict[AgentID, StepOutcome]: ...
```

### 11.2 Built-in Environments

- **TicTacToe** (`environments/tic_tac_toe/`): Two-player game
- **GSM8K** (`environments/gsm8k/`): Math word problems
- **DatasetQAEnv**: Generic one-shot QA from samples

---

## 12. Inference System

### 12.1 Token-In API

```python
# Old (drift-prone)
resp = client.complete(ChatCompletionRequest(messages=...))

# New (drift-free)
template_result = chat_template.apply(messages, tools=...)
resp = client.complete_tokens(TokenCompletionRequest(
    prompt_token_ids=template_result.prompt_token_ids,
    ...
))
```

### 12.2 Weight Synchronization

For online RL, push updated weights to vLLM:

```python
class ChatClient(Protocol):
    def sync_weights(
        self,
        params: Mapping[str, Tensor],
        version: Optional[str | int] = None,
    ) -> str: ...
```

Uses **NCCL** for efficient broadcast from trainer to inference workers.

---

## 13. Key Technical Decisions

### 13.1 Truncation Semantics

Three distinct concepts:
1. **terminated**: Environment reached terminal state
2. **truncated**: Episode cut off by time limit (env or protocol)
3. **llm_finish_reason**: Model stopped generating (`stop`, `length`, `tool_calls`)

Incomplete completions (`finish_reason="length"`) are treated as **parse failures**, not truncation.

### 13.2 Turn-Concatenated Training

One **SAWItem per agent turn**, not per step:
- Stitch rollout-time token traces from each AgentStep
- Interpreter outputs are prompt tokens (masked out)
- Credit from final step in turn

### 13.3 Why Not Value Functions?

Ludic uses **return-based** methods (GRPO, REINFORCE) rather than actor-critic:
- No value network to train
- Group normalization provides baseline
- Simpler for research iteration

---

## 14. Public Interfaces

### 14.1 High-Level Training

```python
from ludic.training import (
    Trainer, TrainerConfig,
    make_grpo, GRPORequestStrategy,
    RolloutBatchSource, RolloutEngine,
)

algo = make_grpo(group_size=4)
engine = RolloutEngine(env_registry, protocol_registry)
batch_source = RolloutBatchSource(engine, ...)

trainer = Trainer(
    model=model,
    algo=algo,
    batch_source=batch_source,
    cfg=TrainerConfig(
        lr=1e-5,
        micro_token_budget=32768,
        max_seq_len=4096,
        sync_every_steps=1,
    ),
    publisher=BroadcastPolicyPublisher(...),
)

await trainer.train(num_steps=1000)
```

### 14.2 Agent Interaction

```python
from ludic.agents import Agent, ToolAgent
from ludic.inference import VLLMChatClient, HFChatTemplate
from ludic.context import FullDialog

agent = Agent(
    client=VLLMChatClient(...),
    model="meta-llama/Llama-3.1-8B",
    ctx=FullDialog(system_prompt="You are helpful."),
    parser=xml_parser("move"),
    chat_template=HFChatTemplate(tokenizer),
)

agent.on_env_reset(obs, info)
result = await agent.act(inference=InferenceSpec(...))
```

### 14.3 Protocol Execution

```python
from ludic.interaction import SingleAgentProtocol

protocol = SingleAgentProtocol(
    agent=agent,
    stop_on_parse_error=False,
    external_tool_handler=my_handler,
)

rollouts = await protocol.run(
    env=my_env,
    max_steps=50,
    inference=InferenceSpec(sampling=SamplingConfig(temperature=0.7)),
)
```

---

## 15. Extensibility Points

| Component | How to Extend |
|-----------|---------------|
| **Algorithm** | New `CreditAssigner` + `Loss`, compose via `RLAlgorithm` |
| **Agent** | Subclass `Agent`/`ToolAgent`/`ReActAgent` |
| **Context** | Implement `ContextStrategy` protocol |
| **Environment** | Implement `LudicEnv`/`SingleAgentEnv` |
| **Protocol** | Implement `InteractionProtocol` |
| **Batch Source** | Implement `BatchSource` protocol |
| **Logger** | Implement `TrainingLogger` protocol |

---

## 16. What Makes Ludic Unique

1. **Multi-step reasoning first**: Built for agentic behavior, not single-turn chat
2. **Full reasoning traces**: Training sees all tool calls and internal steps
3. **Token-in architecture**: No drift between training and inference
4. **Algorithm composition**: Credit + Loss = Algorithm
5. **Clean separation**: Agent ≠ Env ≠ Protocol
6. **Memory efficiency**: SharedContext, selective_log_softmax
7. **Distributed-aware**: FSDP2, NCCL weight sync, DCP checkpointing
8. **Research-grade**: Hackable, not production-hardened

---

## 17. Limitations and Future Work

From the README and CONSIDERATIONS.md:

1. **No value functions**: Pure return-based methods only
2. **Append-only context required**: Turn-concatenation breaks with context truncation
3. **No Gym-style registry**: Envs/agents built on the fly
4. **Experimental pipeline RL**: Actor/learner split via Redis is early stage
5. **No on-policy distillation yet**: Planned feature

Planned additions:
- Single Stream Policy Optimization (SSPO)
- Delegating Protocol for hierarchical agents
- Gym-style registries for agents/envs/protocols

---

## 18. File Reference

### Core Types
- `src/ludic/types.py` - TokenTrace, AgentStep, EnvironmentStep, Rollout

### Agents
- `src/ludic/agents/base_agent.py` - Agent base class
- `src/ludic/agents/tool_agent.py` - ToolAgent with tool calling
- `src/ludic/agents/react_agent.py` - ReActAgent with think-tool-act loop

### Context
- `src/ludic/context/base.py` - ContextStrategy protocol
- `src/ludic/context/full_dialog.py` - FullDialog implementation

### Environments
- `src/ludic/envs/env.py` - LudicEnv, SingleAgentEnv interfaces
- `src/ludic/envs/dataset_qa_env.py` - DatasetQAEnv for QA tasks

### Interaction
- `src/ludic/interaction/base.py` - InteractionProtocol
- `src/ludic/interaction/single_agent.py` - SingleAgentProtocol
- `src/ludic/interaction/multi_agent.py` - MultiAgentProtocol

### Inference
- `src/ludic/inference/client.py` - ChatClient protocol
- `src/ludic/inference/vllm_client.py` - VLLMChatClient implementation
- `src/ludic/inference/chat_template.py` - HFChatTemplate

### Training
- `src/ludic/training/algorithm.py` - RLAlgorithm and presets
- `src/ludic/training/loss.py` - All loss functions
- `src/ludic/training/credit_assignment.py` - Credit assigners
- `src/ludic/training/trainer.py` - Trainer class
- `src/ludic/training/types.py` - SAWItem, SAWBatch
- `src/ludic/training/config.py` - TrainerConfig

### Batching
- `src/ludic/training/batching/engine.py` - RolloutEngine
- `src/ludic/training/batching/synced_batching.py` - RolloutBatchSource
- `src/ludic/training/batching/offline.py` - OfflineBatchSource
- `src/ludic/training/batching/intra_batch_control.py` - GRPORequestStrategy

### Distributed
- `src/ludic/distributed/interfaces.py` - PolicyPublisher
- `src/ludic/distributed/broadcast.py` - BroadcastPolicyPublisher

---

*This analysis was generated from a comprehensive review of the Ludic codebase.*
