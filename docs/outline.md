# Building RLHF from Scratch

## Introduction
- Why build from scratch (deep understanding vs using TRL)
- What I set out to do (reproduce Stiennon et al.)
- Brief overview of the 3-phase pipeline
- What was NOT a goal (equally as important)

## Design and Implementation Decisions
- Paper Selection
- Implementation Design
- On Using Language Models as Development Tools
- Didn't look at reference implementation until evaluation

## The hard stuff - Challenges and Lessons (and what I demonstrated) [Each a subset #1-4]

### Why is KL not zero at init when it should be? 
- Challenge (high level)
- The Symptom(s)
- The Process
- The Bug
- The Lesson

### State-Action Alignment [Skills #1, #4]
- Challenge (high level)
- The Symptom(s)
- The Process
- The Bug
- The Lesson

### Masking - And all the bugs it can create [Skills #1, #2, #4]
- Challenge (high level)
- The Symptom(s)
- The Process
- The Bug
- The Lesson

### Policy Degradation [Skills #2, #3]
- Challenge (high level)
- The Symptom(s)
- The Process
- The Bug
- The Lesson

### Scaling [Skill #3] (possibly break into sub challenges)
Memory
- Challenge (high level - memory harder than time)
- The symptoms (simply OOM, Long run times -> debugging takes too long, failed runs, etc)
- The Primary Causes (scaling response length, many models in working memory, float32, etc)
- The solutions / lessons (building profiling, checkpointing, alerting tools)
Compute Time / $$

### (Maybe more bugs, tbd)

## Results summary + eval/validation [Skill #5]
- Important curves, validation metrics, maybe one example
- Link to eval writeup for more details. 

## Reflections + lessons
- Some numbers on timeline / hours spent / compute time+$ spent, etc
- What I learned - combining did well + can improve 
    - invest in iteration speed, 
    - assume bug before hyperparam, but assume lack of algorithm understanding before bug. "The _hardest to find_ bug is usually in your understanding, not your implementation"
    - even hyperparams can be treated as a bug
    - simple baseline pros / cons
    - think more and type less
    - WIP (come up with more)

##  Future Work / Directions
- Things I want to do but didn't do (implement ZeRO from scratch)
- DPO, RLAIF, other types of RLHF
- Connect to alignment
- Expand to some new ideas
- Implement the GPT from scatch (but maybe copy weights - training would be $$$)