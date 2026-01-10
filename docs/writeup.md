# RLHF from Scratch

Work in LLMs is cool. I wanted to learn more about research that is important today, bridge some of my previous experience in RL, and work on something that easily branches into safety and alignment spaces. 


## Intro - 
*Why build from scratch*, you ask? Great question. Implementing reinforcement learning is notoriously hard, and some might say that "writing things from scratch is the most catastrophically self-sabotaging thing you can do" [[2]](REFERENCES.md#ref2). The goal was simply to have learn and have fun. It is easy to skim a paper and say that you know how it works, but the active learning from struggling and debugging, and satisfaction of understanding from first principles are entirely different when building from scratch. I also enjoy working on the problems found in the bridge between research and practical implementation.

*What was not a goal?* I want to quickly acknowledge that if the immediate goal was to get validate a research idea, this is not the approach I would take unless critically necessary. The time and effort it takes to implement from scratch is at direct odds with the initial need to rapily invalidate or gather supporting evidence for a research idea. Repurposing or extending an existing, battle tested library or repo would be a better way to go. 


## [Before] Project Design and Implementation Decisions
### What paper should I choose?
Since there is no single source of truth paper for RLHF, there were a lot of options of papers to pick [[3]](REFERENCES.md#ref3)[[4]](REFERENCES.md#ref4)[[5]](REFERENCES.md#ref5)[[6]](REFERENCES.md#ref6)[[7]](REFERENCES.md#ref7)[[8]](REFERENCES.md#ref8). I decided on [[1]](REFERENCES.md#ref1), as the authors include significant details, which eliminates a lot of ambiguity from both training and evaluation.  

### Code System Design
Given my background in engineering, I often find research code to be less than elegant. RL systems are particularly difficult, as they are more challenging to simplify and abstract [[2]](REFERENCES.md#ref2). So, a goal of mine was to *make the code largely readable though design rather than comments*. I wanted to be able to look at the code for the PPO algorithm and for to feel as readable as pseudocode, and to feel like I can recursively zoom in on a detail by looking at a function definition and read it like pseudocode. 

### On Using Language Models as Development Tools

I'm dedicating space to this because LLMs are and will continue to be part of the development process for engineers, but there is the very real concern (especially in elementary education) that people let LLMs do think for them. 

**How I used LLMs:**
The goal was to use LLMs for as tools, rather than them using me as a tool. Here are some examples of ways that I used LLMs:
- [Feedback] I'm considering designing a profiling tool as a decorator, but on initial thought it is hard to track state over multiple or recursive calls. Is there a paradigm I haven't thought of that would allow for a better design?
- [Teacher] What is the difference between reserved and allocated GPU memory?  
- [Search/Documentation] How do the HF Llama models when used as a sequence classifier extract the reward? Can you find me the source code that can verify your answer?
- [Intern/Basic Code] Can you write me a few lines of code showing me how I would structure and send a request to the Claude API?
- [Basic Debugging] I have this stack trace and error message I don't recognize or understand, what could be the issue?
- [Idea Generation / Rubber Ducking] My best idea for why this function is taking forever was GPU/CPU I/O, but it looks like its this tensor operation which makes no sense. Can you give me some ideas that could explain these observations?
- [Editor] Can you make this sentence / paragraph sound better?
- [Encouragement] Tell me I'm pretty.

**How I _didn't_ use LLMs**
- [Copy+Paste/Delegation] Can you implement the PPO training loop for me?
- [Complex Debugging] Here is all my code, now tell me why my reward isn't going up.


**What I learned about LLM-assisted development:**
*Okayyyy* I will admit that at times I got lazy and tried the things I said I didn't do. But, I often found that using LLMs in this way was actively detrimental. 
- When LLMs tried to write too much code for you, I often find many more bugs. Even if it was bug free, I often found it to be structured in a way that I thought was bad.
- When dumping info into an LLM and asking it to do complex debugging it could sometimes point out basic errors say within a function (maybe, "you flipped this sign") but it struggled more to reason about complex interactions in the system (cant do "you made this choice here, and it looks fine in these intermediate steps, but that ultimately affects this thing way over here")
- Similarly when I found myself letting it lead the debugging directions, it often ran in circles, chased things down rabbit holes that didn't make sense or were at least not the logic next place to check. Its easy to get in this mode, similar to when its easy to just do a sort of guess and check coding when you wait for the code to compile and run to tell you where the next error is. Instead, its much more efficient to use use your head.

Ultimately, I verified all LLM code generated in the training loops by walking through it myself, and sometimes in eval code I would briefly skim then see if it generated correct output. 

### Don't look at reference implementation until evaluation
I made the choice not to look at reference implementations until evaluation, which is also ill advised [[2]](REFERENCES.md#ref2) but I felt that I would learn more without a cheat sheet. At evaluation, the reference implementation was used maily to identify details that may make direct comparisons different. 

## [During] The hard stuff - fun challenges and technical lessons

### Why is KL not zero before any update step is made? 
**The Symptom** KL divergence was rising quickly during training, but more concerning was that it didn't start at zero. Since the policy and reference models should be identical before any updates, this violated a basic sanity check.

**The Debugging Process** I started with the obvious candidates, like if I was logging KL before the first update step, and if the models were recieving the same inputs. When I visually examined the raw logits, I noticed a significant number with zero probability. From here I realized the model was using top-p sampling and probability masking during generation, but these weren't applied during the forward pass I was using for KL computation.

**The Bug** Generation-time behavior differs from forward-pass behavior due to sampling parameters like `top_p` and `temperature`. I was comparing distributions that weren't actually aligned:
- KL between `policy_logits` and `ref_policy_logits` was computed over different support sets
- Entropy calculations were incorrect for the same reason  
- Training-time forward passes were inconsistent with rollout-time generation

**The Fix** Instead of trying to perfectly replicate generation behavior in the forward pass (controlling for top-p, temperature, masking, etc.), I added a separate forward pass on the generated sequences. While this adds overhead, it's negligible - generation takes ~50x longer due to autoregressive decoding across 50 tokens.


**The Lesson**
WIP


### State-Action Alignment
**The Symptom** This single bug took me the longest to find. PPO training was seeing immediate and accelerating drops in rewards. 

**The Debugging Process** After ruling out all of my initial hypotheses, I was clueless and started debugging by system component. I verified (with lots of analysis) that the reward model (and hence the value) was functioning correctly, and similar for the sft model. I completely eliminated issues with MSE loss / the value function pathway by showing that the policy degraded before the value model was updated, hence it can have no influence on the policy model. I broke down each component of the policy loss (ratios and advantages, which is further broken down into values, rewards, returns, and kl) and inspected them, finding a few small bugs along the way. Once the issue was narrowed down to the GAE calculation (which I had checked multiple times), I eventually realized that there was a discrepancy with what the LLM outputs for each step. 

**The Bug** 
At time t, the models will output the desired state and value. However, the (softmaxed) logits describe the distribution over the actions considered/taken at time t, which is equivalent to the state at time t+1. The advantage is how much better an _action_ is than expectation, it should be aligned to the actions as well. The reward model default outputs how rewareds at EOS tokens, but it describes how good the _action_ of the eos token was, hence rewards are also misaligned.

```
       ----State Indexing---- (len = seq_len)
        Position:            0         1         2        3        4
        states:          [prompt,   token1,   token2,    EOS,     PAD]
        pad_mask:        [   1,        1,        1,       1,       0 ]
        values:          [  V0,       V1,       V2,      V3,      V4 ]

        ----Action Indexing---- (len = seq_len-1)
        Position:            0         1         2        3
        actions:         [token1,   token2,    EOS,     PAD]
        logits:          [   L0,       L1,      L2,      L3 ]
        log_probs:       [  lp0,      lp1,     lp2,     lp3 ]
        rewards:         [    0,        0,      +1,       0 ]
        reward_mask:     [    0,        0,       1,       0 ]
        advantages:      [   A0,       A1,      A2,      A3 ]
        action_pad_mask: [    1,        1,       1,       0 ]
```

**The Lesson** 
- Would unit tests on GAE have caught this? I at least think understanding, coding and testing each essential section independently and more mindfully may have helped  
- Instead of trying to simply translate math to code, thinking more end to end and through what the meaning of the representations of the math and code variables mean would have caught this earlier.


### Masking - And all the bugs it can create [Skills #1, #2, #4]
Masking can easily cause issues everywhere. Recall that we need masking not just for attention in forward passes, but also for padding in pretty much all following computations, as well masking non-reward positions. All functions with any aggregation need to take into account masking - mean, var, softmax, whitening, etc. Additionally, the tricky alignment of state and action tensors mentioned previously means we need be mindful align the masking properly as well. These obsevations apply to both computations in the training loop, as well as all logging statistics.

[TODO] List the symptoms that this can cause


### Scaling - Memory 
- Multiple models / sets of weights in working memory - policy, value, reward, sft, old policy, old value Policy and value need gradients and optimizer states.
- A few massive tensors, particularly after - both policy and sft logits are (batch_size x sequence_length x vocab_size x dtype_size) memory each. 
- Mixed precision
- Gradient accumulation - allows for smaller batches in working memory, but note that this doesn't work for PPO
- Gradient checkpointing - time for memory
- Cleaning tensors in the right places - compared to some single function implementations, garbage collection from abstracted code can help, but around the bottlenecks, strategically ordering operations and deleting tensors can prevent memory spikes
- Memory fragmentation mitigation - helps manage reserved but not allocated memory. Can optimize and coallesce fragments of unallocated, reserved memory into one segment that can be allocated for different use cases at different points in the training loop.
- Tooling - building a profiler that works for memory
- Mindful loading of models with .eval, requires_grad(False), and/or with torch.no_grad to save memory usually occupied by activations, gradients and optimizer states, for instance on SFT and old policy and value models (actually for the latter two, you only need the weight/state dict in memory).
- In place tensor operations 

### Scaling - Time
- Generation is a big bottleneck
- I/O can be a bottleneck when working with distributed training
- Engineering efficiency - startup time, cache data mapping, save models in an efficient way, etc
- Want to try torch.compile()
- Want to try `attn_implementation="flash_attention_2", ` in LlamaForCausalLM.from_pretrained


### Scaling - $$$



### A few more notable ones:


**Mixed Precision (FP16) Causing Infinite Gradients**
Infinite gradients showed up, but were harder to isolate than usual. The forward pass was stable, but backward gradients through the exp() operation in PPO ratio calculation `r = torch.exp(new_log_probs - old_log_probs)` overflowed in FP16, primarily caused by grad scalar- but it really seemed like these numbers weren't that unreasonably big, so I started questioning why FP16 had such a small dynamic range, which led me to discover BF16. 

I learned a lot about how mixed precision training works, led me to think more deeply about different floating point representations


**Policy Degredation**
The policy performed well for about 50 steps, then started quickly degrading. This feels like a classic hyperparameter tuning problem (reduce learning rate?), but I treated it like a bug, isolating where in the system the instability originated and to where it propagated. I eventually found a facepalm bug where I was overloading a parameter name, effectively setting the value coefficient incorrectly. While all the symptoms were correctly indicative of required hyperparameter tuning, the bug was implementation level. I learned (thankfully the easier way) that debugging treating what felt like hyperparameter tuning issues initially as a bug is significantly faster. 






## [After] Results summary + eval/validation 


## Reflections + meta lessons


- Assume a bug rather than hyper parameters
- Invest in accelerating iteration speed; anything so you don't have to run the program from the start, loading models, the dataset, training
    - Caching, building tooling like the profiler, debuggers >>> print statements, 
- Systematic debugging - When problems persist after eliminating all obvious possibilites, something like component or binary search debugging served me well, find a way to test if the bug is in a certain part of the system or occurs before / after a certain place in the training loop. 

- Think first then test - still 
- Unit test critical components (loss function, GAE calculation, etc)
- First, check if what I intended to code is correct - if the math is wrong, it doesn't matter if there is a bug.
- Think of the levels of integration hell

Algorithm --> Your Understanding --> Your intended implementation --> Your actual implementation
- Why ML is hard: https://ai.stanford.edu/~zayd/why-is-machine-learning-hard.html




## Whats next?
- Things I want to do but didn't do (implement ZeRO from scratch)
- DPO, RLAIF, other types of RLHF
- Connect to alignment
- Expand to some new ideas
- Implement the GPT from scatch (but maybe copy weights - training would be $$$)









