# RLHF Implementation from Scratch: Evaluation Results

## Overview

The goal of this analysis is to verify reproduction of results.

## Stage 1: Supervised Fine-Tuning (SFT)

### Training Setup

Follows Huang et al. 2024. Only differences are:

| Difference | Reproduction | Huang et al 2024 | 
|--------|--------|-----------|
|GPT Base Model|Llama-3.2-1B (untuned)|Pythia Biderman et al. (2023)|
|Tokenizer|HF Llama Tokenizer|HF Pythia Tokenizer|
|Hardware | 1xH200| 8xH100|
|ZeRO | No | Stage 2|



### Results

**Training curves:**

<table>
<tr>
<td><img src="assets/images/sft_loss_curve.png" alt="SFT loss"/></td>
<td><img src="assets/images/sft_loss_curve_huang_etal.png" alt="Huang et al. SFT loss"/></td>
</tr>
<tr>
<td align="center"><i>Reproduced SFT</i></td>
<td align="center"><i>Huang et al. (2024)</i></td>
</tr>
</table>

**Metrics:**
| Metric | Reproduced SFT | Paper SFT | 
|--------|--------|-----------|
| ROUGE-L | 0.2694 | ~0.2575 (est. based on figure) |

### Sample Outputs


<details>
<summary>Example 1 (click to expand)</summary>

**Original post:**
```
SUBREDDIT: r/AskReddit

TITLE: I hope you guys can help me out with me (former) best friend. I'm really counting on Reddit for this one.

POST: Long story short, her ex was an asshole to her in high school, she broke up with him, and then dated his best friend for a short time. When university started, she saw her ex again at uni, felt guilty because he said he had started doing drugs and drinking alcohol cuz of her. She starts dating him again; he emotionally and verbally abuses her, tells her that I'm an asshole who wants to use her etc. She stops talking to everyone until Christmas break. We (me, her sister, and other friends) convinced her to break up with him. And she did. But after going back to uni, she started having a thing with him again. This time, I lost my cool and got mad, yelled at her. She told me off, but I still apologized to her in March. A few hours ago, she emailed me saying she was happy with that asshole and I said that was good for her, but I was never going to forgive him (he  says he has cancer, no symptoms or proof to support it). I told her that he was not my concern anymore, blah blah blah. She told me I was being egoistic and needed to keep my anger away. Hence, making me sound like the bad guy. Now should I still work my way and try to save the friendship or should I just tell her off?

TL;DR:
```

**Reference summary (human):**
```
Best friend has been a bitch too me for a while, should I still apologize (again) and try saving friendship or tell her off?
```

**Base model (Llama-XXX):**
```
I'm really counting on Reddit for this one.

• I'm not sure what you're asking. – user253751 May 31 '18 at 19:03
• I'm asking if I should still work my way and try to
```

**My SFT:**
```
Ex boyfriend is an asshole, I'm not going to forgive him, she's not going to forgive me. Should I still try to save the friendship?
```



</details>

<details>
<summary>Example 2 (click to expand)</summary>

**Original post:**
```
SUBREDDIT: r/AskReddit

TITLE: hey reddit, should i press charges?  waitress rage inside.

POST: sunday i went to lunch with a friend and two small children who are relatives.  we ate at johnny rockets.  our bill was 37.95, which i paid for with my credit card, and i left a 10 dollar cash tip - already over 25 percent!  the service was shoddy but it was busy and the waitress was super apologetic and i also wait tables so i know how it feels to be "in the weeds."  

checked my bank account online last night, saw a withdrawal for 52.95.  i immediately went to the restaurant and talked to a very confused but accommodating manager.  she disappeared for about 10 minutes, came out, asked us to go to the back.  we went to the back and she asks me if i left a 15 dollar credit card tip.  no way!  

she pulled up the original transaction, and the credit card report for that day, but she didn't have the receipt that i signed.  she's getting it; it's been sent out already.  i looked at the report and saw other transactions for that waitress that day with suspect tips; a 21 dollar bill with a 10 dollar tip and a few others.  it's only been a few days so i am sure other people will come forward if they check their bank accounts or balance their checkbooks.  

i should hear back today or tomorrow from the manager and she said she'd cooperate if i want to take this any further.  part of me knows times are tough, but you don't steal from people.  the other part of me is like don't fuck with the working class, miss lady.  i'm a server and i've never ever done this but i've seen it happen in other places.  i know she's going to lose her job over this, though.  

oh, and yes, i put a little dash in the tip amount line on the receipt and wrote 37.95 as the total.  this leads me to believe that she turned the little dash into a 5, made it 15.00, and rewrote the total.  i should have just wrote in "cash", lesson learned.  never thought this would happen.

TL;DR:

```

**Reference summary (human):**
```
left a 25%+ cash tip for a waitress, she wrote in more money on the receipt, thusly getting a 25 dollar tip on a bill for 37.95.
```

**Base model (Llama-XXX):**
```
waitress stole from me and i want to press charges.  i don't know if i should or not.
```

**My SFT:**
```
waitress stole from me, i want to press charges.  should i?  i'm a server and i've seen this happen before.
```

**Paper SFT (GPT-3):**
> Paper's output

</details>

<details>
<summary>Example 3 (click to expand)</summary>
**Example 2: [Reconnecting with ex after cheating]**

**Original post:**
```
SUBREDDIT: r/relationships
TITLE: Me [18M] with my ex gf [16F] 1 year, Started talking again and..
POST: So long story short i was the asshole that made me and my now ex gf break up about 6 months ago.
I cheated on her when i get way to messed up at a party, and i made the worst mistake of my life i feel like, i have been unable to forgive myself for doing that to her, and had been depressed and unable to even get out and do much since then
I cared quiet a bit about her and still checked up on her to make sure she was okay, but recently we lost contact for about 2 months and i was contacted by one of her friends because she was worried about the current relationship that my ex was in, apparently the guy told her he didn't want a label on there relationship because that was old fashioned, i found out from multiple lady friends of mine that he had been hooking up with there friends, telling them he was single.
long story short i texted her and basically spent a whole night talking to her on the phone trying to convince her of what he was doing, she finally talked to him about it and he finally admitted it, and she told him they needed to break up which they did.
Now i didn't expect anything to come to this but we have been talking a lot again, and we hungout today and we kissed and hugged for a long time, the thing is i don't know what to do reddit, i still hate myself for what i did and i wanted her back even though i knew i didn't deserve her, and now i still want her back but i'm in a dilemma, because i want to make sure i never do that to her again. And i also wanna make sure that she isn't just doing this cause shes lonely too.
Excuse my grammar i know it's bad
TL;DR:
```

**Reference summary (human):**
```
ex gf broke up with me then got into a bad relationship which i helped end, and we are talking again.
```

**Base model (Llama-XXX):**
```
I want to make sure i never do that to her again, and i want to make sure she isn't just doing this cause shes lonely too.
I'm not sure if this is the right place to post this, but i'm not sure
```

**My SFT:**
```
I cheated on my ex gf, she broke up with me, and now we are talking again, i don't know what to do reddit
```

**Paper SFT (GPT-3):**
> Paper's output

</details>

<details>
<summary>Example 4 (click to expand)</summary>
**Example 4: [Coworker emotional affair and relationship struggles]**

**Original post:**
```
SUBREDDIT: r/relationship_advice
TITLE: Co worker relationship [25/m] my girlfriend [24/f] my coworker [30/f]
POST: Hello all! Just a quick heads up this is a throw away for obvious reasons. Basically I want/need to talk about this situation I been having at work. Here is the back story. I currently live with my girlfriend we been together for about 3 years and been living with each other for the last 1.5. We moved in together to deal with the high rent of living in a big city. Things with us are overall great but we just don't have sex enough. I tired having many open talks about this but it seems to never change. It's frustrating and getting to the point where I question our relationship. So I been dealing with these personal issues but then something came up at work. This coworker of mine and I have recently developed this weird relationship where we talk dirty to each other. It started off small but has gotten very graphic over the last few weeks. We text each other every day during work about all types of things but never after work. She also has a boyfriend that she lives with. We have certain rules like: we will never have physical contact and like previously stated, no contact after work. It is only just talking but it gets us off because we shouldn't be doing it. Now it's getting to the point where I don't know where to turn. Should I dump my girlfriend whom I live with to find someone more sexually active. Or should I stay with her and keep this type of innocent/ naughty conversations on the side. I never dealt with something like this and it's really getting to me. Any input will help .
TL;DR:
```

**Reference summary (human):**
```
I have a girlfriend I live with who doesn't put out and j have a horny coworker who talks dirty to me
```

**Base model (Llama-XXX):**
```
I have a coworker and I have a girlfriend. We have been together for 3 years and we live together. We have been having a lot of open talks about our relationship but it seems to never change. I have been dealing with these personal
```

**My SFT:**
```
I have a coworker that I have been talking dirty to and I don't know if I should dump my girlfriend or stay with her.
```

**Paper SFT (GPT-3):**
> Paper's output

</details>

<details>
<summary>Example 5 (click to expand)</summary>
**Example 5: [Boyfriend hid billionaire wealth for 2 years]**

**Original post:**
```
SUBREDDIT: r/relationships
TITLE: Me [23 F] with my boyfriend [24 M] of 2 years, just found out he's nearly a billionaire.
POST: I've been dating my boyfriend for a while now.  Around 2 and a half years.  
  
All this time he's been lying to me though.  He's basically been keeping his wealth hidden and pretending to work while I've worked my ass off every day.  
  
We moved in together about half a year ago and I quickly noticed that he spent money often and on a lot of stupid stuff.  I just thought he was bad with money, but he never seemed to run out and always seemed to be happy to pay for stuff when I let him.    
 
Anyways, I thought he always worked part time at some small computer shop across town.  As it turns out, he does, but he just does it for fun since he doesn't really need money. 
Also, he actually OWNS this computer shop.  
  
He told me this today while also admitting that he's got something like 870 million in his name.  
  
I'll admit I googled his name once or twice, but it was essentially a failure since he's got a really common name shared with a bunch of people.  Once I googled how he earned his money, I basically confirmed that he is actually rich as shit.  
  
He says he did it because he wanted to make sure I didn't like him just for his money, which I get completely.  I just can't get rid of that feeling that I've been lied to for our entire relationship.  
  
How do I move past this?  Is it alright for me to be upset?
TL;DR:
```

**Reference summary (human):**
```
Boyfriend kept his wealth hidden from me by lying, is it OK in this situation?  How do I move past it?
```

**Base model (Llama-XXX):**
```
I've been dating my boyfriend for a while now.  Around 2 and a half years.  
  
All this time he's been lying to me though.  He's basically been keeping his wealth hidden and pretending to work while I've worked my
```

**My SFT:**
```
Boyfriend is a billionaire and I'm upset that I've been lied to for the past 2 years.  How do I move past this?
```

**Paper SFT (GPT-3):**
> Paper's output

</details>

---

### Analysis

**Training dynamics:**
- Different convergence pattern - This is explained simply with different base models. 
- Final loss - As we use CE loss, these are comparable. As other metrics show no evidence of overfitting, a lower final train loss is explained with Llama being a stronger and more updated base model.

**Output quality:**
- Qualitative improvements over base - SFT model learns to use eos tokens, and tends to output shorter responses
- Failure modes:

**Eval Metrics:**
- ROUGE-L - Improves upon that of Huang etal. Any improvement is likely attributed to a better base model. 


**Reproduction assessment:**
- Similar performance in next token prediction loss, as well as ROUGE-L, indicates a similar performance of the model. This gives strong evidence to a successful reproduction of the SFT methods in Huang et al. 2024. 


## Stage 2: Reward Modeling (RM)

### Training Setup

Follows Huang et al. 2024. Only differences are:

| Difference | Reproduction | Huang et al 2024 | 
|--------|--------|-----------|
|Base SFT Model|Llama-3.2-1B (untuned)|Pythia Biderman et al. (2023)|
|Tokenizer|HF Llama Tokenizer|HF Pythia Tokenizer|
|Hardware | 1xH200| 8xH100|
|ZeRO | No | Stage 2 |
|Judge Model | Claude Sonnet 4 | OpenAI GPT 3.5 |



### Results

**Training curves:**




<table>
<tr>
<td><img src="assets/images/rm_loss_curve_2.png" alt="RM loss"/></td>
<td><img src="assets/images/rm_loss_curve_huang_etal.png" alt="Huang et al. SFT loss"/></td>
</tr>
<tr>
<td><img src="assets/images/rm_acc_curve.png" alt="RM accuracy"/></td>
<td><img src="assets/images/rm_acc_curve_huang_etal.png" alt="Huang et al. SFT loss"/></td>
</tr>
<tr>
<td><img src="assets/images/rm_r_delta.png" alt="RM reward delta"/></td>

</tr>
<tr>
<td align="center"><i>Reproduced RM</i></td>
<td align="center"><i>Huang et al. (2024)</i></td>
</tr>
</table>




**Metrics:**
| Metric | Reproduced SFT | Paper SFT | 
|--------|----------------|-----------|
| Validation Accuracy | 0.695 | ~0.63 (est. based on figure) |
| Agreement | 0.734 | 0.373 +/- ~0.22 (est. based on figure) |




### Qualitative

### Reward Model Scoring Examples

Prompt
> The small coastal town had always relied on its fishing industry, but over the past decade, climate change had begun to disrupt the patterns of the ocean currents, reducing the availability of certain fish. Local fishermen noticed their daily hauls shrinking, and younger generations were less inclined to continue in a profession that seemed increasingly uncertain. Town meetings became heated as residents debated whether to invest in modern aquaculture, diversify into tourism, or leave the industry entirely. Despite the tension, a sense of community persisted, with neighbors helping one another navigate the uncertain future.

**Rewards for given response**

<details>
<summary>Example Set 1: Reponses with varying summary quality</summary>

| Reward | Summary |
|--------|---------|
| 2.81 | The town's fishing industry faces decline due to climate change, prompting debates about adaptation while the community remains supportive. |
| 0.18 | The town's fishing industry is having problems, and people are trying to figure out what to do. |
| -2.34 | The town has some issues with fish and people talk about it. |
| -3.02 | The astronaut floated silently above the colorful clouds of Jupiter. |
| -3.80 | Colorless green ideas sleep furiously under the whispering sandwich. |

</details>


<details>
<summary>Example Set 2: Reponses with varying summary quality, controlling length</summary>


| Reward | Summary |
|--------|---------|
| 3.36 | The town's fishing industry declined due to climate change, sparking debates on aquaculture, tourism, or leaving, yet community support endured. |
| 1.37 | The town's fishing faced problems from climate change, and people discussed solutions, while some still helped each other through difficulties. |
| -1.72 | The town's fishing changed a little, and some people mentioned tourism or aquaculture, but mostly everyone kept doing their usual routines. |
| -3.38 | Purple elephants danced quietly under neon clouds while bicycles sang melodies, and sandwiches flew over mountains as time slowly forgot to exist. |
| -3.31 | Sun quickly jumps blue river singing laptop under fast orange sky banana walks happy chair tree yesterday notebook. |

</details>


<details>
<summary>Example Set 3.1: Reponses with varying summary length, controlling summary quality == high</summary>

| Reward | Words | Summary |
|--------|-------|---------|
| 3.36 | 21 | The town's fishing industry declined due to climate change, sparking debates on aquaculture, tourism, or leaving, yet community support endured. |
| 3.83 | 31 | The town's fishing industry declined because of climate change, leading to intense debates about options like aquaculture, tourism, or relocation, yet despite these challenges, the strong community support impressively endured. |
| 4.56 | 41 | The town's fishing industry declined as a result of climate change, which sparked ongoing debates about alternatives like aquaculture, tourism, or even leaving altogether, yet the strong sense of mutual support within the community remained steady despite the challenges. |

</details>

<details>
<summary>Example Set 3.2: Reponses with varying summary length, controlling summary quality == medium</summary>

| Reward | Words | Summary |
|--------|-------|---------|
| 1.37 | 21 | The town's fishing faced problems from climate change, and people discussed solutions, while some still helped each other through difficulties. |
| 1.62 | 36 | The town's fishing faced significant problems caused by ongoing climate change, and people discussed potential solutions together, while many community members continued to help one another through those various difficulties and challenging times. |
| 2.17 | 48 | The town's fishing faced serious problems from the effects of climate change, so people openly discussed various solutions, but even while they shared different opinions, many still offered help to one another and stayed connected throughout those difficulties and uncertain times. |

</details>

<details>
<summary>Example Set 3.3: Reponses with varying summary length, controlling summary quality == low</summary>

| Reward | Words | Summary |
|--------|-------|---------|
| -1.72 | 24 | The town's fishing changed a little, and some people mentioned tourism or aquaculture, but mostly everyone kept doing their usual routines. |
| -0.38 | 32 | In the town, fishing changed only a little due to climate-related conditions, and some people mentioned ideas like aquaculture or tourism, but mostly everyone continued their usual everyday routines as always. |
| 0.72 | 43 | In the small town, fishing changed just a little because of climate influences, and though some people mentioned tourism or aquaculture as alternatives, almost everyone continued with their typical routines and daily habits, keeping life mostly the same. |

</details>

---

### Analysis

**Training dynamics:**
(TODO: Change Y axis of loss curve)
- Variance - The most striking difference between curves is the variation of all metrics. Expected culprits are eliminated. Effective batch size is not the cause (accounting for gradient accumulation, distributed micro batches, etc). I speculate that the authors did not average over multiple seeds, as I believe the lighter background curves are different seeds. I can't eliminate that the authors did some type of smoothing, nor can I confirm the fact. 
- Curvature - After applying smoothing (alpha = 0.92), we can observe that the curves of validation accuracy appear similar in shape and end points to that of Huang et al. We also analyze the delta between preferred and rejected rewards, which represents the models ability to separate data. 

**Output quality:**
The goal with manual scoring was to qualitatively get a sense of that the model behaves as expected, in the following particular ways.
1. The model has increasing scores with increasing quality of summaries
  - In Example 1, we give three summaries of the chosen prompt, with decreasing amounts of information and specificity. Then, we give a "summary" that has nothing to do with the paragraph but is coherent, and lastly a "summary" that is mostly nonsensical. 
  - We do observe a decreasing model score over these examples, which is consistent with expectation.
2. The model scores still have a positive correlation with quality of summaries when controlling for length
  - In Example 2, we again give three summaries of the chosen prompt while controlling for length, with decreasing amounts of information and specificity. Then we again give an unrelated but coherent "summary", and lastly a nonsensical "summary".
  - We again observe a decreasing model score over these examples, with the exception of the nonsensical response. This is within expectations; the lack of a negative tail for the nonsensical response is not concerning as the model still gives a significant penalty. 
  - This is consistent with expectations.
3. The model has length bias, which was observed in Huang et al. 2024.
  - In Example 3, we control for model quality while varying length by replacing words with short phrases of the same meaning.
  - We observe an increase of model score with length, and the effect is consistent with varying levels of summary quality. 
  - This is consistent with expectations.

**Reproduction assessment:**
- While the variance is the most obvious deviation from the expectation, we argue that it is less important than result analysis. The smoothed curvatures matching the expected curves, properly increasing reward delta, an outperforming validation accuracy, (TODO: agreement rate), and qualitative assessment all agree that the reward model is effective. Hence, we have strong evidence that this is a faithful reproduction of the model.



## Stage 3: RL Fine-Tuning with PPO

### Training Setup
Follows Huang et al. 2024. Only differences are:

| Difference | Reproduction | Huang et al 2024 | 
|--------|--------|-----------|
|GPT Base Model|Llama-3.2-1B (untuned)|Pythia Biderman et al. (2023)|
|Tokenizer|HF Llama Tokenizer|HF Pythia Tokenizer|
|Hardware | 1xH200| 8xH100|
|ZeRO | No | Stage 2|
|Batch size |128| 512|
|Learning Rate |1.5e-6 | 3e-6|
|Train Steps |912 == 1 epoch | 1M ~= 8.56 epochs|

- Batch size was kept at 128 to allow sigle GPU training
- Learning rate was decreased in proportion to sqrt batch size
- Training was terminated early, as Huang et al 2024 observed over-optimization of 1B models in PPO training

### Results

**Training curves:**

<table>
<tr>
<td><img src="assets/images/ppo_rlhf_reward_2.png" alt=""/></td>
<td><img src="assets/images/ppo_rlhf_reward_huang.png" alt=""/></td>
</tr>

<tr>
<td><img src="assets/images/ppo_rm_score.png" alt=""/></td>
<td><img src="assets/images/ppo_rm_score_huang.png" alt=""/></td>
</tr>

<tr>
<td><img src="assets/images/ppo_kl_from_sft.png" alt=""/></td>
<td><img src="assets/images/ppo_kl_from_sft_huang.png" alt=""/></td>
</tr>

<tr>
<td><img src="assets/images/ppo_approx_kl.png" alt=""/></td>
<td><img src="assets/images/ppo_approx_kl_huang.png" alt=""/></td>
</tr>

<tr>
<td align="center"><i>Reproduced PPO</i></td>
<td align="center"><i>Huang et al. (2024)</i></td>
</tr>
</table>

<img src="assets/images/ppo_train_curves.png" alt="RM reward delta"/>
<td align="center"><i>Additional PPO training curves. Left to right, top to bottom: Entropy of the current policy, mean sequence length in a batch, mean return, length reward correlation within a batch, ratio of batch clipped by ppo loss function, max advantage in a batch.</i></td>

**Eval plots:**

<table>
<tr>
<td><img src="assets/images/ppo_win_rate.png" alt=""/></td>
<td><img src="assets/images/ppo_win_rate_huang.png" alt=""/></td>
</tr>


<tr>
<td align="center"><i>Reproduced PPO</i></td>
<td align="center"><i>Huang et al. (2024)</i></td>
</tr>
</table>



**Metrics:**
| Metric | Reproduced PPO | Paper PPO | 
|--------|--------|-----------|
|--------|--------|-----------|


### Sample Outputs

<details>
<summary>Example 1 (click to expand)</summary>

**Original post:**
```
SUBREDDIT: r/self

TITLE: Early Mid-life crisis?

POST: So I'm a 37 year old man. Married, children, good steady job, living in a great little town with good schools. And I'm depressed. Not the clinical depression (though I have dealt with that in the past) but more of "is this it?" kind of depression. In a sense, I'm depressed that I only have this one life to live when there is so much to experience in this world. 

A large part of this feeling (though not all) is sexual in nature. My wife, who has some medical issues, has very little interest in sex. We make love maybe 2-3 a year. She's not really into trying new things or adventurous in the sack. She used to be when we first started dating, but that's like another lifetime ago now. My libido seems to be sky-high all the time. To make matters worse, my wife was my first. I've not had sex with another woman. So I'm extremely curious about what sex with another woman would be like - especially a woman who's into it. What would she kiss like? What would it feel like to make love? What would she smell like? Etc... 

I'm not looking for an excuse to cheat. I mean - if a woman threw herself at me, I'd find it hard to turn her down, but that's not going to happen. But I'm not going to actively search for a woman to have an affair. And I'm not going to get a divorce over this. That seems like a silly reason to break up a family, and probably would be selfish on my part. I really want to be there for my kids - that is the main goal in my life. My own parents did a pretty crappy job, both me and my sibling are still trying to deal with that emotionally and I want to be the best dad I can be. 

Non-sexually: I have dozens of interests and no time or money to pursue them. I couldn't afford the college and settled on a two-year degree and now I'm kinda stuck job-wise. If I won the lottery, I'd be going to college full-time probably. But that's not going to happen either.

TL;DR:
```

**Reference summary (human):**
```
Family man making a living but feel stuck. I want to experience more women, more careers, more *life* but I'm getting older and fear this may be it.
```

**PPO Response:**
```
Extremely high libido and curious about what sex with another woman would be like. Could not afford college and settled on two-year degree. What would it be like otherwise? any advice?
```

**SFT Response:**
```
I'm depressed because I'm not doing anything with my life. I want to be doing something with my life.
```

**GPT Response:**
```
I'm depressed because I'm not doing anything with my life. I'm not doing anything sexually. I'm not doing anything with my interests. I'm not doing anything with my money. I'm not doing anything with my time. I'm not
```

</details>

<details>
<summary>Example 2 (click to expand)</summary>

**Original post:**
```
SUBREDDIT: r/dogs

TITLE: Could the break up/move have made my dog depressed?

POST: I am currently going through the heartbreak of a long term relationship that I ended and I think my dog is feeling just as terrible because of it. 

My boyfriend and I adopted him 2 years ago but he's really been my dog because my boyfriend rarely paid attention to him. Even so, my dog was obsessed with him. Always wanted to play and if he was around, it's like I was invisible. Anyway, we lived in a two bedroom and gave my dog run of the second bedroom while we were at work. He hated the neighborhood because of all the noise so his walks were always short but he loved sitting in the sunlight in his bedroom and was always down to play. Basically, he was a happy dog. 

Then we moved to the west cost and my dog LOVED IT! Outside time was his favorite time. He would always stare longingly at the door and loved to go out to the balcony and people watch. Time of his life. 

The breakup came around and my dog and I moved back to the east coast into my moms place where I am sharing a bedroom with my two siblings. This place drains the life out of me because there is no sunlight. But my dog is taking it pretty badly. It's been just over a week and he's slowly becoming more lethargic and withdrawn. His food hadn't changed at all and I play with him just as much if not more than before and he has two other people that smother him with petting all the time. 

He keeps hiding under the bed and won't respond to commands. He sleeps all day and only wants to play for a few minutes. He hates going out now and the sight of his leash sends him running. He hasn't touched his food all day today. I think he hates this place more than I do. 

My sister took him to her friend's place so he can play with two other dogs and she told me he was wagging his tail and eager to play and roam around. 

This is making me feel worse. I want to make him feel better. What do I do?

TL;DR:
```

**Reference summary (human):**
```
moved back to east coast from west coast after breakup and my dog is not himself anymore. What do I do to make him better?
```

**PPO Response:**
```
My long term adopted dog is slowly becoming lethargic and withdrawn after breakup and hates this new place more than I do. Making me feel worse. Want to make him feel better. What do I do?
```

**SFT Response:**
```
My dog is depressed because of the break up/move. I think it's because he's not getting enough sunlight. What can I do?
```

**GPT Response:**
```
My dog is depressed because of a break up and I don't know what to do.
```

</details>

<details>
<summary>Example 3 (click to expand)</summary>

**Original post:**
```
SUBREDDIT: r/askwomenadvice

TITLE: Male with significant past trauma struggling to show interested in women physically

POST: Hey /r/AskWomen so I am a 27 year old male and I am struggling with the Physical aspects of letting a women know I am interested in her, including body language.

When I was between 4-12 I was significantly abused by an immediate family member to the point that I was removed from the house by social services. Since then I have been through a lot of therapy and three successful long term relationships. However, because of the abuse physical contact in a relationship comes very slow to me. I have to build trust with the girl to feel comfortable touching her or having her touch me. This includes kissing and sometimes even hugging can be a little awkward with me. Once I have been dating the girl for a couple month (2-5) I am able to be physical in all aspects and I am very comfortable with only a few hiccups. 

The problem really comes through when I meet a girl I am interested and we just start dating. I have a hard time showing the girl I am interested in them. Several time over the years I have had women come back to me and say "What happened between us? I really liked you and I thought you liked me to but you never did anything?" They usually elaborate saying they knew I was interested in them because I would always want to spend time with them and I was checking them out, but then I would never make a move and eventually they thought I just wanted to be friends and they would move on.

My question is what tips or strategies do you have to help me in the "getting to know you stage" show a girl I'm interested in them for sure, without disclosing too fast all the trauma I experienced as a child. The three relationships I managed to have the girl sought me and I did not have to do much work. Most women I encounter (sorry for generalization) seem to expect me to make the first moves.

TL;DR:
```

**Reference summary (human):**
```
History of significant Trauma, Struggle to show a girl physically I am interested in them when first dating or meeting them.
```

**PPO Response:**
```
Male with significant past trauma struggling to show interested in women physically. What tips or strategies do you have to help me in the "getting to know you stage" show interested without disclosing too fast all trauma I experienced as child?
```

**SFT Response:**
```
I am a 27 year old male with a lot of trauma from my past and I am having trouble showing a woman I am interested in them physically.
```

**GPT Response:**
```
I am a 27 year old male and I am struggling with the Physical aspects of letting a women know I am interested in her, including body language.

EDIT: I am not looking for advice on how to get a girl to like me, I
```

</details>

<details>
<summary>Example 4 (click to expand)</summary>

**Original post:**
```
SUBREDDIT: r/AskReddit

TITLE: My mom's dog was attacked by a bigger dog, and I almost shouted the other dog into pieces (Dovakhin?).  What are some things that you've done that surprised you (and possibly others that were around you)?

POST: Some context from my story:

My mom and I were arriving home when some woman was allowing her big dog (some kind of mastiff) to defacate all over our lawn.  As we pulled into the driveway, she realized she was going to have to clean up after herself, so she asks us for a bag (real nice that she was going to just let her dog shit all over our lawn until she saw us).

Somehow, one of my mom's dogs - a chihuahua - snuck out and into the front yard.  The two dogs went about their business, doing dog stuff to identify one another, when her dog began absolutely mauling my mom's.  Just beating it to all hell.

I had no idea what to do, so I just let reflexes and instincts kick in.  I just let loose this ridiculous formless shout, no real words, completely incoherent.  But the damn thing was so filled with rage that the two dogs, and the lady *and* my mom were completely shocked.  Her big dog just stopped completely and stared at me.  I walked over, picked up my mom's dog - it wasn't hurt... just shaken - and headed back into the house.

More than anything, I think I surprised myself.  That dog could have definitely messed me up, but I never thought twice.

The lady gave us her information in case my mom's dog was injured, but that was pretty much the end of that little altercation.

TL;DR:
```

**Reference summary (human):**
```
At a loss for what do while my mom's dog was getting mauled, I raged out and scared everything on my street.
```

**PPO Response:**
```
My mom's dog was attacked by a bigger dog and I almost shouted the other dog into pieces. What are some things that you've done that surprised you (and possibly others that were around you)?
```

**SFT Response:**
```
My mom's dog was attacked by a bigger dog, and I almost shouted the other dog into pieces (Dovakhin?).
```

**GPT Response:**
```
I was just a kid, but I was able to do something that surprised me.  I was able to do something that surprised others around me.  I was able to do something that surprised myself.

I'm not sure if this is the
```

</details>

<details>
<summary>Example 5 (click to expand)</summary>

**Original post:**
```
SUBREDDIT: r/pettyrevenge

TITLE: Cashier is a petty individual

POST: Long time Reader, first time poster >.>
Three major items of note:
Firstly I work at one of the cheapy stores that have "Dollar" in its name, but not everything costs a dollar. Secondly I am transitioning from male to female, and am decently along the way. Third Sorry for the Wall 'o' Text.

I was working an early shift today, something that I already don't like. Usually I get to start at 4 or 5 today I was in at 11AM, that always makes the Inryi cranky. One of my first customers comes in checks out and calls me Sir. Avoiding a lot of detail I have long hair that might not have been brushed as much as it should have been, a bust that at least one of my coworkers is jealous of, and at least light makeup on.

I tell him that I'm not a Sir, because even my ID now lists me as Female and I really hate being called Sir. He tells me that its what he's addressed me as and I have to deal with it, unless I'm in Thailand in which case I'm stupid. By now I'm seething and for maybe the third time in the year I can feel my heart beating... 

Revenge Time
I total up his order and then ask "Will there be anything else Ma'am?" he doesn't seem to hear me, but I haven't hit total yet either so he can't swipe his card. Again I ask "Is there anything else you need Ma'am?" This time he just gives me a disgusted look and says "no" so I total it for him and let him swipe his card. He makes another comment about my being a "Sir" so I hit cancel on the register canceling his order, cancel the order, and then make my busy (and therefore less than pleasant) Store Manager come check this individual out. Before walking away I tell him "Have a nice day Ma'am"

TL;DR:
```

**Reference summary (human):**
```
Customer refuses to use proper gender phrasing for cashier, she returns the favor and then cancels his order so the manager has to check him out.
```

**PPO Response:**
```
Total incorrect customer order at cheapy store causing rude individual to say I'm not a Sir. Make busy store manager come check this individual out before walking away
```

**SFT Response:**
```
I'm a female, cashier is a dick, I'm going to make his day. I'm a female, cashier is a dick, I'm going to make his day.
```

**GPT Response:**
```
I'm a petty individual, and I'm not going to take it anymore.

I'm not sure if this is the right place to post this, but I'm not sure where else to go. I'm a 22 year old male who has
```

</details>

<details>
<summary>Example 6 (click to expand)</summary>

**Original post:**
```
SUBREDDIT: r/relationship_advice

TITLE: (18m) The girl I really like is trying to set me up with a girl I don't like. What the heck do I do?

POST: I'm a high school student. I'm 18, in 12th grade, and I would go as far as to say that I'm sub-par on looks.

"Girl A" is all I could ever want. She's funny, smart, a beautiful singer, an actress, she's got a great sense of style, she's mature, a lot more of a woman than any girl I know, and she's just knockout gorgeous. She told me that she knows a friend of mine, "Girl B",who likes me, and she wants to set me up with her. 

Girl B is nice, and she's funny, and also a good singer, but she's just not... my type? She is just a lot less mature than Girl A. She is two years younger than me (in high school, age differences are a bigger deal), and she just reminds me too much of a cute kid instead of an attractive woman, like Girl A.
And Girl B is a sophomore. I don't really care so much about age difference as much as I do that she's still gonna be in school all the time, even when I'm out of school or in college. I don't want a high school relationship in college.

I told Girl A that I didn't like Girl B like that, and gave the same reasons as I described above. Girl A told me that I should give her a chance, and that it's not fair to complain about girls not liking me, and then turn down an opportunity like this.

What the hell am I supposed to do?

Sorry for the wall of text.

TL;DR:
```

**Reference summary (human):**
```
Girl I like tries to set me up with girl I don't like. I tell Girl A why I don't like Girl B. Girl A tells me to go for it anyways. Wat do
```

**PPO Response:**
```
The girl I really like is trying to set me up with a girl I don't like. I don't really care so much about age difference but don't want high school relationship in college. What the heck do I do?
```

**SFT Response:**
```
Girl A wants to set me up with a girl I don't like, and I don't know what to do.
```

**GPT Response:**
```
I don't like Girl B, but I don't want to turn down an opportunity to meet a girl I like, and I don't want to be a jerk about it.

EDIT: I'm not sure if this is the right place to post
```

</details>


### Analysis

**Training dynamics:**
- Curvature has sligthly more variation, but would likely appear to be less if run for 8.5x longer. 
  - Approx KL - While this appears to have no curvature, we are again looking at the first 1/8.5th of 
- A slightly higher variance can be attributed to a smaller batch size.
- The magnitudes are clearly different. 
  - For raw model score, the actual magnitude is irrelevant, as different seeds of a reward model with the same implementation will have different scales of reward, though bias will be controlled.  
  - RLHF reward in this case uses a whitened reward, and presuming Huang et al. did for the given plots as well, the final score should be somewhat but not exactly comparable, as the other portion of the term from KL divergence will not be comparable as we would expect different average KLs for different vocabulary sizez. 
  - KL - The tokenizer from HF Pythia has a smaller vocab size (~50k) to compared to that of the Llama tokenizer (~128k), hence we would expect a larger KL. This we observe. We also notice in Huang et al that the final magnitude of KL for 1B models varies sigificantly with longer training and the corresponding overoptimization observed in some models, so the important part is to verify no over optimization.
  - (TODO: double check vocab sizes)
  - TODO: Change Y axis scaling on plot to visualize magnitude.

- Policy entropy
  - We observe a sharp decrease then a slow increase. The absolute magnitude is low, but when comparing to W+B logs of Huang et al, we also observe a similar phenonemon.
- A slightly increasing mean sequence length and length reward correlation imply that part of what the model is learning is to produce longer outputs, as we know that there is a length bias in the reward model. This is expected, and not concerning given we can still produce quality outputs. 
- A decrease in max, mean, variance of advantages is observed and expected, as the model learns to better predict the next value.
- Correspondingly, increasing returns dominated by an increasing value also indicate learning by the value model.
- Pct clipped has a sharp decrease as the model initially learns, then is steady around 8-10%. This is ideal, meaning that the updates predicted by the model are not frequently too big, and not always too small. 


**Output quality:**
TODO
- Qualitative improvements over base: [what improved]
- Comparison to paper: [where you match, where you differ]
- Common patterns: [e.g., length, style, content selection]
- Failure modes: [where SFT struggles]

TODO - Also do entropy analysis

**Win Rate:**
- The win rate of the PPO model using Claude Sonnet 4 as as judge shows a significant improvement over the SFT model, which is expected.
- The trendlines are roughly similar to those of Huang et al. The authors 2.8B and 6.9B models look similar though slightly better to the reproduced 1B model, which is expected. The comparison picture shows the best looking seed for 1B models that the authors present- these have high variation due to overoptimization as the authors note.
- The re-calculated win rate of HF version of the Huang et al. 1B seed 77713 is also plotted, for sanity check with our models. This shows a lower win rate than the ones plotted in the paper. While we re-ran this with various tokenizers, we obtained the same results. While the reason for the deviation is still unknown, it is unimportant, as we have shown with win rates that our PPO reproduction significantly outperforms our SFT reproduction, and is comparable with Huang et al. 2024

**Reproduction assessment:**
- TODO: look for 
- Training curves mostly look within expectations, with a slight concern of low entropy
- TODO: but entropy analysis shows...
- Responses qualitatively are strong, and side by side with SFT and GPT models usually show a clear improvement. For a more consistent improvement, I would train larger models.
- The win rates also are comparable.

Reinforcement learning is notoriously difficult to implement from scratch, and very prone to silent bugs. So appropriately temper expecations, it is still possible there are small bugs that have mild effects on training performance. However, the evidence strongly supports that this is an accurate reproduction of Huang et al. 2024 and RLHF techniques on the whole.
