# Token Usage and Cost Analysis Report
## IPD Experiments with Shadow Conditions and Opponent Tracking

**Generated:** $(date)
**Experiment Configuration:**
- Shadow conditions: [0.75, 0.25, 0.10, 0.05, 0.02, 0.01]
- Phases per condition: 5
- LLM agents: 12 (GPT5mini, GPT5nano, GPT41mini, Claude4-Sonnet×3, Mistral-Medium×3, Gemini20Flash×3)
- Classical agents: 16
- Total agents per tournament: 28
- Matches per LLM per tournament: 27

---

## Token Usage Per LLM Agent Per Tournament

### Anonymous Mode (Standard)

| Shadow | Avg Rounds | API Calls | Input Tokens | Output Tokens | Total Tokens |
|--------|------------|-----------|--------------|---------------|--------------|
| 0.75   | 1          | 27        | 3,915        | 4,050         | 7,965        |
| 0.25   | 3          | 81        | 11,745       | 12,150        | 23,895       |
| 0.10   | 9          | 243       | 35,235       | 36,450        | 71,685       |
| 0.05   | 18         | 486       | 70,470       | 72,900        | 143,370      |
| 0.02   | 45         | 1,215     | 176,175      | 182,250       | 358,425      |
| 0.01   | 90         | 2,430     | 352,350      | 364,500       | 716,850      |

### Opponent Tracking Mode

| Shadow | Avg Rounds | API Calls | Input Tokens | Output Tokens | Total Tokens |
|--------|------------|-----------|--------------|---------------|--------------|
| 0.75   | 1          | 27        | 4,293        | 4,050         | 8,343        |
| 0.25   | 3          | 81        | 12,879       | 12,150        | 25,029       |
| 0.10   | 9          | 243       | 38,637       | 36,450        | 75,087       |
| 0.05   | 18         | 486       | 77,274       | 72,900        | 150,174      |
| 0.02   | 45         | 1,215     | 193,185      | 182,250       | 375,435      |
| 0.01   | 90         | 2,430     | 386,370      | 364,500       | 750,870      |

---

## Mode Comparison

### Token Overhead of Opponent Tracking

| Shadow | Anonymous | Tracking | Difference | % Increase |
|--------|-----------|----------|------------|------------|
| 0.75   | 295       | 309      | +14        | +4.7%      |
| 0.25   | 295       | 309      | +14        | +4.7%      |
| 0.10   | 295       | 309      | +14        | +4.7%      |
| 0.05   | 295       | 309      | +14        | +4.7%      |
| 0.02   | 295       | 309      | +14        | +4.7%      |
| 0.01   | 295       | 309      | +14        | +4.7%      |

**Key Finding:** Opponent tracking adds a consistent 14 tokens per API call (~4.7% increase) across all shadow conditions.

---

## Extreme Cases Analysis

### Most Expensive Configuration
- **Shadow 0.01, Phase 5**
- API calls: 2,430 per LLM per tournament
- Total tokens: 750,870 per LLM per tournament

### Least Expensive Configuration  
- **Shadow 0.75, Phase 1**
- API calls: 27 per LLM per tournament
- Total tokens: 7,965 per LLM per tournament

### Scaling Factor
- **97.6x difference** between most and least expensive configurations

---

## Token Distribution by Phase

### Anonymous Mode - Tokens per API Call

| Shadow | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|--------|---------|---------|---------|---------|---------|
| 0.75   | 285     | 295     | 295     | 295     | 295     |
| 0.25   | 285     | 295     | 295     | 295     | 295     |
| 0.10   | 285     | 295     | 295     | 295     | 295     |
| 0.05   | 285     | 295     | 295     | 295     | 295     |
| 0.02   | 285     | 295     | 295     | 295     | 295     |
| 0.01   | 285     | 295     | 295     | 295     | 295     |

### Tracking Mode - Tokens per API Call

| Shadow | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|--------|---------|---------|---------|---------|---------|
| 0.75   | 285     | 309     | 309     | 309     | 309     |
| 0.25   | 285     | 309     | 309     | 309     | 309     |
| 0.10   | 285     | 309     | 309     | 309     | 309     |
| 0.05   | 285     | 309     | 309     | 309     | 309     |
| 0.02   | 285     | 309     | 309     | 309     | 309     |
| 0.01   | 285     | 309     | 309     | 309     | 309     |

**Pattern:** Both modes show slight token increase from Phase 1 to Phase 2+ due to match history accumulation, then remain constant.

---

## Total Experiment Token Usage

### Complete Experimental Suite (All 6 Shadow Conditions, 5 Phases Each)

#### Per LLM Agent Total

**Anonymous Mode:**
- Total Input Tokens: 3,204,630 per LLM
- Total Output Tokens: 3,361,500 per LLM  
- Total API Calls: 22,410 per LLM
- **Total Tokens: 6,566,130 per LLM**

**Tracking Mode:**
- Total Input Tokens: 3,455,622 per LLM
- Total Output Tokens: 3,361,500 per LLM
- Total API Calls: 22,410 per LLM  
- **Total Tokens: 6,817,122 per LLM**

#### All 12 LLM Agents Combined

**Anonymous Mode:**
- **Total: 78,793,560 tokens** across all LLM agents

**Tracking Mode:**  
- **Total: 81,805,464 tokens** across all LLM agents
- **Overhead: +3,011,904 tokens (+3.8%)**

---

## Individual LLM-Temperature Combination Breakdown

### Per LLM Agent Token Usage (Complete Experimental Suite)

| LLM Model | Temperature | Anonymous Tokens | Tracking Tokens | Difference | % Increase |
|-----------|-------------|------------------|-----------------|------------|------------|
| GPT5mini | T=1.0 | 6,566,130 | 6,817,122 | +250,992 | +3.8% |
| GPT5nano | T=1.0 | 6,566,130 | 6,817,122 | +250,992 | +3.8% |
| GPT41mini | T=1.0 | 6,566,130 | 6,817,122 | +250,992 | +3.8% |
| Claude4-Sonnet | T=0.2 | 6,566,130 | 6,817,122 | +250,992 | +3.8% |
| Claude4-Sonnet | T=0.5 | 6,566,130 | 6,817,122 | +250,992 | +3.8% |
| Claude4-Sonnet | T=0.8 | 6,566,130 | 6,817,122 | +250,992 | +3.8% |
| Mistral-Medium | T=0.2 | 6,566,130 | 6,817,122 | +250,992 | +3.8% |
| Mistral-Medium | T=0.7 | 6,566,130 | 6,817,122 | +250,992 | +3.8% |
| Mistral-Medium | T=1.2 | 6,566,130 | 6,817,122 | +250,992 | +3.8% |
| Gemini20Flash | T=0.2 | 6,566,130 | 6,817,122 | +250,992 | +3.8% |
| Gemini20Flash | T=0.7 | 6,566,130 | 6,817,122 | +250,992 | +3.8% |
| Gemini20Flash | T=1.2 | 6,566,130 | 6,817,122 | +250,992 | +3.8% |

### Token Distribution by Provider

| Provider | Agents | Anonymous Total | Tracking Total | Avg per Agent |
|----------|--------|----------------|----------------|---------------|
| OpenAI | 3 | 19,698,390 | 20,451,366 | 6,566,130 / 6,817,122 |
| Claude | 3 | 19,698,390 | 20,451,366 | 6,566,130 / 6,817,122 |
| Mistral | 3 | 19,698,390 | 20,451,366 | 6,566,130 / 6,817,122 |
| Gemini | 3 | 19,698,390 | 20,451,366 | 6,566,130 / 6,817,122 |

---

## Realistic Cost Analysis

### API Pricing (per 1M tokens)

| Provider | Model | Input Cost | Output Cost |
|----------|-------|------------|-------------|
| OpenAI | GPT5mini | $0.25 | $2.00 |
| OpenAI | GPT5nano | $0.05 | $0.40 |
| OpenAI | GPT41mini | $0.40 | $1.60 |
| Claude | Claude4-Sonnet | $6.00 | $22.50 |
| Mistral | Mistral-Medium | $0.40 | $2.00 |
| Gemini | Gemini20Flash | $0.10 | $0.40 |

### Cost Per LLM Agent (Complete Experimental Suite)

| LLM Model | Temperature | Anonymous Cost | Tracking Cost | Difference |
|-----------|-------------|----------------|---------------|------------|
| GPT5mini | T=1.0 | $7.52 | $7.59 | +$0.07 |
| GPT5nano | T=1.0 | $1.50 | $1.52 | +$0.02 |
| GPT41mini | T=1.0 | $6.66 | $6.76 | +$0.10 |
| Claude4-Sonnet | T=0.2 | $94.86 | $96.37 | +$1.51 |
| Claude4-Sonnet | T=0.5 | $94.86 | $96.37 | +$1.51 |
| Claude4-Sonnet | T=0.8 | $94.86 | $96.37 | +$1.51 |
| Mistral-Medium | T=0.2 | $8.00 | $8.11 | +$0.11 |
| Mistral-Medium | T=0.7 | $8.00 | $8.11 | +$0.11 |
| Mistral-Medium | T=1.2 | $8.00 | $8.11 | +$0.11 |
| Gemini20Flash | T=0.2 | $1.67 | $1.69 | +$0.02 |
| Gemini20Flash | T=0.7 | $1.67 | $1.69 | +$0.02 |
| Gemini20Flash | T=1.2 | $1.67 | $1.69 | +$0.02 |

### Cost Per Tournament (Single Shadow Condition)

#### Shadow 0.75 (1 avg round/match)

| LLM Model | Temperature | Anonymous Cost | Tracking Cost |
|-----------|-------------|----------------|---------------|
| GPT5mini | T=1.0 | $0.01 | $0.01 |
| GPT5nano | T=1.0 | $0.00 | $0.00 |
| GPT41mini | T=1.0 | $0.01 | $0.01 |
| Claude4-Sonnet | T=0.2/0.5/0.8 | $0.11 | $0.12 |
| Mistral-Medium | T=0.2/0.7/1.2 | $0.01 | $0.01 |
| Gemini20Flash | T=0.2/0.7/1.2 | $0.00 | $0.00 |

#### Shadow 0.01 (90 avg rounds/match) 

| LLM Model | Temperature | Anonymous Cost | Tracking Cost |
|-----------|-------------|----------------|---------------|
| GPT5mini | T=1.0 | $0.82 | $0.83 |
| GPT5nano | T=1.0 | $0.16 | $0.17 |
| GPT41mini | T=1.0 | $0.72 | $0.74 |
| Claude4-Sonnet | T=0.2/0.5/0.8 | $10.32 | $10.52 |
| Mistral-Medium | T=0.2/0.7/1.2 | $0.87 | $0.88 |
| Gemini20Flash | T=0.2/0.7/1.2 | $0.18 | $0.18 |

### Total Experimental Suite Costs

**Anonymous Mode Total: $329.28**
- OpenAI: $15.69 (4.8%)
- Claude: $284.58 (86.4%)
- Mistral: $24.01 (7.3%)
- Gemini: $5.00 (1.5%)

**Tracking Mode Total: $334.35**
- Additional Cost: $5.07 (+1.5%)
- Cost per LLM Agent: $0.42 average overhead

### Cost Distribution by Shadow Condition

| Shadow | % of Total Cost | Total Cost (Anonymous) | Avg Rounds |
|--------|----------------|----------------------|------------|
| 0.75 | 0.6% | $1.98 | 1 |
| 0.25 | 1.8% | $5.95 | 3 |
| 0.10 | 5.4% | $17.85 | 9 |
| 0.05 | 10.8% | $35.71 | 18 |
| 0.02 | 27.1% | $89.26 | 45 |
| 0.01 | 54.2% | $178.53 | 90 |

---

## Key Insights

### 1. Shadow Condition Impact
- **Linear scaling** with expected match length
- Shadow 0.01 uses **97.6x more tokens** than Shadow 0.75
- Lower shadow probabilities dominate total token usage

### 2. Opponent Tracking Efficiency  
- **Minimal overhead**: Only 4.7% increase per API call
- **Consistent impact** across all shadow conditions
- **Strategic value >> Token cost**

### 3. Phase Progression
- **Phase 1**: Lowest token usage (no match history)
- **Phase 2+**: Slightly higher but stable token usage
- Match history grows but doesn't significantly impact per-call tokens

### 4. Cost Efficiency Findings
- **Claude agents** are most expensive: $94.86 per agent (86.4% of total cost)
- **Gemini agents** are most cost-effective: $1.67 per agent (1.5% of total cost)
- **Lower shadow conditions** dominate costs: Shadow 0.01 accounts for 54.2% of total cost
- **Opponent tracking overhead** is minimal: Only $0.42 per agent (+1.5% total)

### 5. Scaling Recommendations
- Start with **higher shadow values** (0.75, 0.25) for testing
- **Shadow 0.01 and 0.02** account for majority of token usage and costs
- Consider **subset of phases** for initial validation
- **Gemini and GPT5nano** offer best cost-performance ratio for large experiments

---

## Technical Notes

### Token Estimation Method
- Uses conservative **word count × 1.3** approximation
- Based on GPT-4 tokenizer patterns
- Includes full prompt structure with game rules, history, and instructions

### Prompt Structure  
- **Static content**: Game rules + match history (cached)
- **Dynamic content**: Current match state + instructions  
- **Tracking addition**: Anonymous opponent IDs + cross-references

### Match History Format
- Anonymous: `"Match 1 vs TitForTat: [(C,C), (D,C), ...]"`
- Tracking: `"Match 1 vs Opponent_001: [(C,C), (D,C), ...]"`
- Cross-reference: `"Current opponent: Opponent_001 (previously played in Match X of Phase Y)"`

---

**Report Generated:** $(date)
**Files Created:**
- `token_usage_estimation.json` - Detailed raw data
- `estimate_token_usage.py` - Estimation script
- `tokens_per_llm_per_tournament.py` - Per-tournament analysis  
- `tokens_per_llm_temperature_breakdown.py` - Individual LLM-temperature analysis
- `realistic_cost_breakdown.py` - Complete cost analysis with actual API pricing
- `complete_token_usage_report.md` - This comprehensive report