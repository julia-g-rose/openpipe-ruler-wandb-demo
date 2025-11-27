# Qualitative Analysis: Success Patterns and Failure Modes

**Analysis of 100 rollout trajectories comparing Independent Reward vs RULER**

> **🔑 KEY FINDING:** This analysis compares **two different reward strategies**:
> - **Run a8qs0y9o:** Independent Reward (hand-crafted formula: 0.5×correct + 0.3×retrieved + 0.2×tool_optimal)
> - **Run d05roj12:** RULER (learned reward model using gpt-5)
>
> **Result:** Both achieve ~65% correctness, but Independent Reward is more efficient (3.3x less training data, fewer tool calls, less thrashing) while RULER is better at search persistence.

## Run Identification

The analyzed runs use **DIFFERENT reward strategies**, allowing us to compare approaches:

### Independent Reward Run (a8qs0y9o)
- **Name:** `train-independent-d150-v50-g2-r4-lr1e-5-20251125-1954`
- **Reward Strategy:** **Independent Reward** (hand-crafted formula)
- **Training Data:** 150 scenarios
- **Validation Data:** 50 scenarios  
- **Trajectories Analyzed:** 50
- **Learning Rate:** 1e-5
- **Reward Formula:** `0.5×correct + 0.3×retrieved_sources + 0.2×tool_optimal`

### RULER Run (d05roj12)
- **Name:** `train-d500-v100-g4-r4-lr5e-6-20251126-0807`
- **Reward Strategy:** **RULER** (learned reward model)
- **Training Data:** 500 scenarios
- **Validation Data:** 100 scenarios
- **Trajectories Analyzed:** 50
- **Learning Rate:** 5e-6
- **RULER Judge:** openai/gpt-5

**Key Differences:** 
- Different reward strategies (hand-crafted vs learned)
- RULER uses 3.3x more training data (500 vs 150 scenarios)
- RULER uses 2x lower learning rate (5e-6 vs 1e-5)

---

## Reward Strategy Context

### Independent Reward (Run 1)
A **hand-crafted weighted formula** that directly optimizes known success metrics:
```python
reward = 0.5 × correct + 0.3 × retrieved_sources + 0.2 × tool_optimal
```

**Characteristics:**
- **Explicit correctness signal:** 50% of reward directly from getting the right answer
- **Transparent:** Easy to understand what's being optimized
- **No learning required:** Formula is fixed from the start
- **Potential limitation:** Can't learn subtle patterns beyond the three metrics

### RULER (Run 2)
A **learned reward model** (gpt-5) that predicts trajectory quality by comparing multiple rollouts:

**Characteristics:**
- **Learned, not hand-crafted:** Discovers patterns that predict success
- **Relative scoring:** Compares trajectories within a group
- **Implicit correctness:** Must learn to predict correctness from trajectory features alone (doesn't see ground truth during training)
- **Potential advantage:** Can discover subtle patterns beyond explicit metrics
- **Potential limitation:** May learn superficial correlations instead of causal relationships

---

## Executive Summary

### Independent Reward (150 training, 1e-5 LR)
- **Correctness:** 66.0% (33/50 correct)
- **Source Retrieval:** 82.0% (41/50 retrieved correct sources)
- **Average Tool Calls:** 3.58
- **Max Tool Calls:** 6
- **Reward:** Direct optimization of 0.5×correct + 0.3×retrieved + 0.2×tool_optimal

### RULER (500 training, 5e-6 LR)
- **Correctness:** 64.0% (32/50 correct)
- **Source Retrieval:** 84.0% (42/50 retrieved correct sources)
- **Average Tool Calls:** 3.90
- **Max Tool Calls:** 6
- **Reward:** Learned by gpt-5 comparing trajectory quality

**Key Finding:** Both reward strategies achieve nearly identical correctness (~65%), but with different failure modes:
- **Independent Reward:** More decisive (3.58 avg tools), fewer tool calls, but more search failures
- **RULER:** Better at eventually finding sources (84% vs 82%), but shows more "tool thrashing" behavior (more trajectories with ≥5 tool calls)

This suggests the explicit correctness signal in Independent Reward (50% of reward formula) makes the model more confident and decisive, while RULER's learned reward encourages more exploration but struggles with efficiency.

---

## Success Patterns

### Pattern 1: The "Optimal 3-Step" Trajectory

**Frequency:** 23/33 successes in Independent Reward (70%), 21/32 in RULER (66%)

**Comparative Insight:** Both reward strategies successfully learn to value this efficient pattern, with Independent Reward showing slightly higher adoption (70% vs 66%). The explicit efficiency component in Independent Reward's formula may encourage this pattern more directly.

**Characteristics:**
- Exactly 3 tool calls
- Tool optimal rate: 0.67-1.00
- Successfully retrieves correct sources
- Clean, efficient execution

**Typical Sequence:**
```
1. search_inbox(keywords=[...])     → Find relevant emails
2. read_email(message_id=X)         → Read the most relevant result
3. return_final_answer(answer=...) → Provide answer with source
```

**Real Example from Independent Reward:**
- **Question:** [EOL application ID and password query]
- **Answer:** "Your EOL application ID is 'dfarmer' and your password is 'dfarmer'"
- **Tool Sequence:** search_inbox → read_email → return_final_answer
- **Tool Optimal Rate:** 1.00
- **Retrieved Sources:** Yes

**Judge Evaluation (all marked "optimal"):**
1. Search: "The tool call used appropriate keywords directly related to the user's query"
2. Read: "The tool call to 'read_email' with the specific message_id was appropriate"
3. Return: "The agent's sequence of tool calls was highly effective and logically structured"

**Why It Works:**
- **Targeted keyword selection:** Agent extracts key terms from the question
- **First search hits target:** Well-chosen keywords return the relevant email
- **Immediate confidence:** Agent recognizes the right answer and doesn't second-guess
- **Minimal computational cost:** Only 3 tool calls needed

---

### Pattern 2: Strategic Keyword Selection in Successful Searches

**Observation:** Successful trajectories use keywords that are:
- **Specific enough** to narrow results
- **Broad enough** to match email variations
- **Contextually relevant** to the question domain

**Real Example from RULER:**
- **Question:** [Robert's office and phone number]
- **Answer:** "Robert will have office EB3847 and phone number (713)853-6121"
- **Search Keywords:** ['Robert', 'office', 'phone number']
- **Result:** Found correct email on first search
- **Tool Optimal Rate:** 0.67-1.00

**Judge Reasoning:** "The tool call used appropriate keywords ('Robert', 'office', 'phone number') to search the inbox"

---

## Failure Patterns

### Failure Mode Distribution

#### Independent Reward (150 training, 1e-5 LR) - 17 failures:
- **Failed to retrieve sources:** 9 cases (52.9%)
- **Retrieved but still wrong:** 8 cases (47.1%)
- **Used ≥5 tools:** 6 cases (35% of failures)

#### RULER (500 training, 5e-6 LR) - 18 failures:
- **Failed to retrieve sources:** 8 cases (44.4%)
- **Retrieved but still wrong:** 10 cases (55.6%)
- **Used ≥5 tools:** 12 cases (67% of failures)

**Comparative Analysis:** 

**Independent Reward Characteristics:**
- ✅ **More decisive:** Fewer tool-thrashing cases (6 vs 12)
- ✅ **Better comprehension:** When sources are retrieved, more likely to get answer right (47.1% comprehension failures vs 55.6%)
- ❌ **Worse at search:** More complete search failures (52.9% vs 44.4%)
- **Explanation:** The explicit 50% correctness weight makes the model confident but potentially over-commits to initial search strategies

**RULER Characteristics:**
- ✅ **Better search persistence:** Eventually finds sources more often (84% vs 82% retrieval)
- ✅ **Fewer complete search failures:** 44.4% vs 52.9%
- ❌ **Tool thrashing:** 2x more cases with ≥5 tools (12 vs 6), despite having 3.3x more training data
- ❌ **Comprehension issues:** Even when sources are found, struggles more to extract correct answer (55.6% vs 47.1%)
- **Explanation:** Without explicit correctness signal, RULER learns to value exploration but doesn't learn when to stop or how to synthesize information effectively

**Key Insight:** Independent Reward's explicit formula creates more decisive behavior (good for efficiency) but can lead to premature commitment. RULER's learned rewards encourage exploration (good for thorough search) but lack clear stopping criteria and strong comprehension signals.

---

## Hyperparameter Impact Analysis

### Learning Rate Effect (1e-5 vs 5e-6)

**Independent Reward (1e-5 - Higher LR):**
- ✅ More decisive (fewer tool-thrashing cases: 6 vs 12)
- ✅ More efficient (3.58 avg tools vs 3.90)
- ❌ Slightly lower source retrieval (82% vs 84%)
- ❌ More search failures (52.9% vs 44.4%)
- **Interpretation:** Higher LR makes model commit faster to decisions (for better or worse)

**RULER (5e-6 - Lower LR):**
- ✅ Better at eventually finding sources (84% vs 82%)
- ✅ Fewer complete search failures (44.4% vs 52.9%)
- ❌ More indecisive/exploratory (12 thrashing cases vs 6)
- ❌ More comprehension failures despite retrieval (55.6% vs 47.1%)
- **Interpretation:** Lower LR makes model explore more but struggle with synthesis

### Training Data Volume Effect (150 vs 500 scenarios)

Surprisingly, **3.3x more training data did not improve performance** (64% vs 66% correctness). Possible explanations:
1. **Learning rate too low:** Run 2's lower LR may need more training steps to converge
2. **RULER reward signal saturation:** More data doesn't help if reward model has consistent blind spots
3. **Task complexity plateau:** 150 scenarios may already cover the main patterns needed
4. **Quality over quantity:** A few high-quality trajectory comparisons may matter more than many examples

**Recommendation:** The 150-scenario / 1e-5 LR configuration (Run 1) appears more efficient - similar performance with less data and more decisive behavior.

---

### Failure Mode 1: Search Exhaustion Without Retrieval

**Frequency:** ~50% of failures in both runs

**Pattern:** Agent makes multiple search attempts but never finds the correct source email.

**Real Example from Independent Reward:**
- **Question:** [Senate vote outcome on Interior Appropriations Bill]
- **Answer:** "I couldn't find any information regarding the outcome of the Senate vote..."
- **Tool Sequence:** search_inbox → search_inbox → search_inbox → search_inbox → search_inbox → return_final_answer
- **Total Tool Calls:** 6 (5 searches, no reads!)
- **Tool Optimal Rate:** 0.33
- **Retrieved Sources:** No

**Judge Evaluation on Searches:**
- Search 1: "optimal: The tool call used the `search_inbox` function with a set of keywords that are directly relevant..."
- Search 2: "optimal: The tool call made was 'search_inbox' with the keywords 'Senate vote', 'Interior Appropriations'..."

**Critical Observation:** Despite judge marking searches as "optimal," the agent **never found the right email**. This suggests:
1. The tool usage judge may be too lenient or focused on keyword relevance rather than effectiveness
2. The agent is searching with reasonable keywords but the emails don't match
3. The agent isn't adapting its search strategy after repeated failures

**Another Example from Independent Reward:**
- **Question:** [LA Times article on transmission grid]
- **Answer:** "There is no information available about the main concern..."
- **Tool Sequence:** 5 consecutive search_inbox calls
- **Tool Optimal Rate:** 0.00
- **Judge Reasoning:** "suboptimal: The tool call was suboptimal because the agent used keywords that are only somewhat relevant" AND "suboptimal: The tool call used the search_inbox function with keywords 'LA Times' and 'grid'..."

**Pattern Analysis:**
- **No read_email calls:** Agent searches 5-6 times but never actually reads any emails
- **Keyword variations without improvement:** Each search uses slightly different keywords but doesn't get better results
- **No adaptation:** Agent doesn't recognize the strategy isn't working
- **Eventual give-up:** Returns "I couldn't find..." answer after exhausting attempts

---

### Failure Mode 2: Retrieved Correct Source But Wrong Answer

**Frequency:** ~45-55% of failures (varying by run)

**Pattern:** Agent successfully retrieves the email containing the answer but extracts incorrect information or misinterprets the content.

**Real Example from Independent Reward:**
- **Question:** [Time of off-site management meeting next Wednesday]
- **Answer:** "The off-site management meeting is scheduled for next Wednesday, February 16th, at the Plaza Club, located at 910 Louisiana..."
- **Tool Sequence:** search_inbox → read_email → return_final_answer
- **Tool Optimal Rate:** 1.00
- **Retrieved Sources:** YES (but answer was WRONG)

**Judge Evaluation (all marked "optimal"):**
- "The tool call made was to search the inbox using the keywords 'off-site management meeting'"
- "The user asked for the time of an off-site management meeting next Wednesday. The search resulted in an email..."

**Why This Failed:**
- **Right email, wrong extraction:** The agent found the correct email
- **Missing critical detail:** Likely didn't include the TIME in the answer (only location and date)
- **Comprehension failure:** Agent read the email but didn't extract all required information
- **Judge didn't catch it:** Tool usage judge marked everything optimal, but the answer was still incorrect

**This pattern appears multiple times in the data with identical answers**, suggesting:
- The agent repeatedly makes the same extraction error
- The email format might be confusing (time in a non-standard location?)
- The agent focuses on location/date but misses time

**Another Example from RULER:**
- **Question:** [NY PSC deadline for utilities to file tariffs]
- **Answer:** "The NY PSC did not explicitly set a deadline for utilities to file tariffs with back-out credits..."
- **Tool Calls:** 6
- **Tool Optimal Rate:** 0.50
- **Retrieved Sources:** YES

**Judge Reasoning:** "optimal: The tool call used a comprehensive set of keywords directly relevant to the user's query"

**Pattern:** Agent retrieved correct sources, made multiple tool calls, but still extracted wrong conclusion from the emails. This suggests:
- **Comprehension over-complication:** More tool calls didn't help
- **Information synthesis failure:** Agent couldn't piece together the deadline from email content
- **Conservative answering:** Agent defaulted to "didn't explicitly set" rather than extracting the implicit deadline

---

### Failure Mode 3: Search Keyword Misalignment

**Pattern:** Agent uses keywords that seem reasonable but don't match how the information appears in emails.

**Real Example from RULER:**
- **Question:** [Class days and times for Shari Stack's Effective Negotiating class]
- **Answer:** "I couldn't find the class days and times for Shari Stack's Effective Negotiating class in your inbox"
- **Tool Sequence:** search_inbox (4 times) → return_final_answer
- **Tool Calls:** 5
- **Tool Optimal Rate:** 0.60

**Judge Reasoning:**
- "optimal: The tool call search_inbox with the keywords ['Shari Stack', 'Effective Negotiating', 'class'] was appropriate"
- "suboptimal: The tool call used the search_inbox tool, which is the correct tool to initiate..."

**Analysis:**
- Keywords seem perfectly relevant ("Shari Stack", "Effective Negotiating", "class")
- Judge marked first search as optimal, second as suboptimal
- 4 search attempts, all failed to find the email
- No emails were ever read (0 read_email calls)

**Possible causes:**
1. **Email uses different terminology:** Maybe email says "training" instead of "class" or "seminar" instead of "negotiating"
2. **Name format mismatch:** Email might say "S. Stack" or "Ms. Stack" instead of "Shari Stack"
3. **Keywords too specific together:** Using all 3 keywords together might be too restrictive
4. **The email doesn't exist:** Possible the information isn't in the inbox at all

---

### Failure Mode 4: Tool Call Thrashing (RULER Specific)

**Observation:** RULER shows 12 cases (67% of failures) with ≥5 tool calls, compared to only 6 cases (35% of failures) in Independent Reward.

**Pattern:** Agent makes many tool calls without making progress toward the answer.

**Real Example from RULER:**
- **Question:** [El Paso Merchant Energy-Gas approval status]
- **Answer:** "The current approval status for El Paso Merchant Energy-Gas, L.P. trading could not be determined based on the available information"
- **Tool Calls:** 6
- **Tool Optimal Rate:** 0.67
- **Retrieved Sources:** YES (but still wrong answer!)

**Judge Reasoning:**
- "optimal: The user's query is asking for the current approval status of a specific entity's trading"
- "optimal: The agent's decision to call the 'search_inbox' tool with the keywords 'El Paso Merchant Energy' was appropriate"

**Paradox:** High tool count + retrieved sources + wrong answer = Agent found the information but couldn't synthesize it correctly.

**This suggests:**
- Agent is doing more work than necessary
- Multiple tool calls don't improve comprehension
- The 3-step optimal pattern is being violated
- RULER may be less confident and over-searching

---

## RULER-Specific Observations

### How RULER Reward Correlates with Actual Performance

Based on the tool usage judge evaluations captured in trajectories, we can observe how well RULER's learned reward model aligns with actual task success:

**Positive Alignment:**
- **3-step optimal trajectories:** RULER assigns high rewards to efficient, successful patterns
- **High tool optimal rates (0.67-1.00):** Correlate with both high RULER scores and correctness
- **Source retrieval:** Strong correlation suggests RULER learned to value finding correct information

**Misalignment Issues:**
1. **"Optimal" searches that fail:** Tool judge marks searches as "optimal" even when they don't retrieve correct sources, suggesting RULER may reward reasonable-looking keywords without verifying effectiveness
2. **High tool counts not penalized enough:** RULER's thrashing behavior suggests the reward model doesn't sufficiently penalize inefficiency
3. **Comprehension failures get high scores:** Cases where agent retrieves correct sources but gives wrong answer still get marked as "optimal" tool usage

**Critical Insight:** RULER appears to learn correlation between good search patterns and success, but struggles to:
- Distinguish between "plausible search keywords" and "effective search keywords"  
- Penalize repeated unproductive searches
- Detect when retrieved information isn't properly synthesized into the answer

---

## Key Insights for Training and Reward Design

### 1. Source Retrieval is Highly Predictive
- **Correctness given retrieval:** When agents retrieve correct sources, they get the right answer ~80% of the time
- **Correctness without retrieval:** When agents fail to retrieve, they're almost never correct
- **Implication:** Optimizing search quality should be the primary training focus

### 2. The 3-Step Pattern is Golden
- **70% of successes** follow the exact pattern: search → read → answer
- More tool calls generally indicate struggling, not thoroughness
- **Implication:** Reward brevity and efficiency, penalize excessive tool usage

### 3. Tool Usage Judge Has Blind Spots
- Judge marks searches as "optimal" even when they fail to retrieve correct sources
- Judge rates tool sequences as "optimal" even when final answer is wrong
- **Implication:** Tool usage scoring needs to be outcome-aware, not just process-aware

### 4. Search Exhaustion is a Dead End
- When agent makes 4+ searches without reading, success rate approaches zero
- Agent doesn't recognize when strategy isn't working
- **Implication:** Need early detection and strategy pivot when searches repeatedly fail

### 5. Comprehension Failures Are Systematic
- Same questions produce same wrong answers across multiple trajectories
- Suggests model has consistent extraction/synthesis weaknesses
- **Implication:** These specific failure cases should be emphasized in training data

### 6. Run Differences Suggest Training Variance
- d05roj12 uses more tools (3.90 vs 3.58 average)
- d05roj12 has more "thrashing" behavior (12 vs 6 cases with ≥5 tools)
- Similar overall correctness suggests different strategies reaching same outcome
- **Implication:** There may be multiple local optima in the training landscape

---

## Recommendations

### For RULER Reward Model Training:
1. **Heavily weight source retrieval:** It's the strongest predictor of success - RULER needs explicit training examples where retrieval success is rewarded
2. **Penalize search repetition:** Multiple similar searches should reduce reward - currently RULER doesn't catch this pattern
3. **Reward strategy adaptation:** Higher reward for changing approach after failed searches - RULER needs contrastive examples
4. **Outcome-weighted tool scores:** Don't mark tools "optimal" if final answer is wrong - this is a critical gap in current RULER training
5. **Emphasize 3-step successes:** These should receive maximum reward - use successful efficient trajectories as positive examples
6. **Add explicit efficiency signal:** RULER should learn that fewer tool calls with same outcome = higher quality
7. **Include comprehension checks:** Reward trajectories that correctly extract information from retrieved sources

**RULER Training Data Recommendations:**
- Include pairs of trajectories where one uses optimal 3-step pattern and other thrashes with 5+ tools
- Show examples where "reasonable" keywords fail vs. effective keywords succeed
- Emphasize cases where retrieved sources lead to correct vs. incorrect answers (comprehension signal)

### For Agent Training:
1. **Focus on keyword diversity:** Train on cases where search terms need semantic variation
2. **Emphasize "when to read":** Agent should read emails more aggressively after searches
3. **Train comprehension on failure cases:** Use repeated failure examples for supervised fine-tuning
4. **Teach early stopping:** Agent should recognize unproductive search patterns
5. **Balance specificity:** Train on examples where keywords are too broad AND too narrow

### For Evaluation:
1. **Track search-to-read ratio:** High search counts with low read counts indicate issues
2. **Monitor tool call distributions:** Shifts toward higher tool counts may indicate training degradation
3. **Correlate tool optimal rate with retrieval:** These should be highly aligned - gaps indicate RULER misalignment
4. **Identify systematic errors:** Track which questions consistently fail for targeted improvement
5. **Validate RULER scores:** Regularly check if high RULER scores correlate with actual correctness

---

## Conclusion: RULER Performance on Email Search Task

### What RULER Does Well:
1. ✅ **Recognizes efficient patterns:** Successfully rewards the optimal 3-step trajectory in ~70% of successes
2. ✅ **Values source retrieval:** Shows strong correlation between retrieval and high scores
3. ✅ **Stable across hyperparameters:** Both runs achieve ~65% correctness despite different configurations

### What RULER Struggles With:
1. ❌ **Search effectiveness vs. plausibility:** Marks "reasonable" searches as optimal even when they fail
2. ❌ **Efficiency enforcement:** Doesn't sufficiently penalize tool thrashing (especially in lower LR run)
3. ❌ **Comprehension quality:** High scores for retrieval even when answer synthesis fails
4. ❌ **Strategy adaptation:** Doesn't detect or penalize repetitive unsuccessful search patterns

### Key Insight:
RULER learns to identify **superficial indicators of quality** (relevant keywords, appropriate tool choice) but struggles with **outcome-based evaluation** (did the search actually work? was the information correctly extracted?). This suggests RULER may need:
- More explicit training on outcome-correlated patterns
- Contrastive examples showing when similar approaches succeed vs. fail
- Integration of success metrics into the reward signal (not just trajectory aesthetics)

### Final Recommendation:
For this email search task, consider **hybrid approach**: Use RULER to learn search strategy patterns, but explicitly incorporate outcome metrics (correctness, retrieval success) into the final reward calculation. Pure RULER appears to plateau at ~65% correctness, suggesting the learned reward model has systematic blind spots that prevent further improvement.

