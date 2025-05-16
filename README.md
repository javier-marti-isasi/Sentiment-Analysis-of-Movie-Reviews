# Sentiment Analysis of Movie Reviews – AI-Powered System

This project delivers an AI-powered solution for **sentiment analysis** on movie reviews, enhanced with **retrieval-augmented generation (RAG)** techniques and **fine-tuning** to improve accuracy and context understanding.

---

## Project Structure

- **`repo/sentiment_analysis_modeling`**  
  Contains:
  - **Task 1**: Model implementation (DistilBERT-based)
  - **Task 3**: Model evaluation and experimentation

- **`repo/sentiment_analysis_api`**  
  Contains:
  - **Task 2**: FastAPI service for inference and retrieval of similar reviews

- **`task_4/`**  
  Contains:
  - **Task 4**: Deployment strategy with architecture, scalability, and cost considerations

Note: Models and datasets are not included in the repository to reduce its size.  
However, they will be automatically downloaded when running the code.  
You can find the complete implementation at the following link:
https://drive.google.com/file/d/1OFoj1enMY0XWfWZZR0MSCWsGPvZhKi8U/view?usp=sharing

---

## Dataset

- **Source**: IMDB movie review dataset
- **Usage**: 
  - 5,000 samples for training  
  - 1,000 for validation  
  - 1,000 for testing  
  This split was used to ensure reliable performance evaluation with limited compute resources.

---

## Experiment Summary (Task 3: Testing & Performance)

Several approaches were tested to evaluate sentiment classification performance:

| Approach                                         | Accuracy |
|--------------------------------------------------|----------|
| Pretrained transformer (DistilBERT)              | 88%      |
| RAG-style retrieval from similar reviews         | 64%      |
| Hybrid model (DistilBERT + retrieval)            | 86%      |
| Finetuned DistilBERT (with limited input length) | 85%      |

- **Selected model for deployment**: `distilbert-base-uncased-finetuned-sst-2-english` (pretrained)  
  Chosen for its excellent **performance–efficiency trade-off**, especially under constraints like limited training time and inference speed in production environments.

---


## Example API input data
```
{
  "review_text": "This movie exceeded my expectations..."
}
```

## Example API output data
```
{
  "sentiment": "positive",
  "confidence": 0.9987996816635132,
  "similar_reviews": [
    {
      "text": "I began watching this movie with low expectations, as a matter of fact i only noticed it because it was an adaptation of a S.K. novel ( a novel i never read).<br /><br />I'm glad my expectations were low because the movie wasn't nothing close to good, but it manages to keep you interested. What really drags this story down is the work done by the director and the actors. The movie is overlong, hasn't no \"nice\" shots and no scares, the dialogs are dumb and the special effects are crap.<br /><br />The only things good are that, as i said, it keeps you interested ( i guess the book must be good) without using much horror cliches.<br /><br />My Vote 4/10.",
      "label": "negative",
      "similarity": 0.22800004482269287
    },
    {
      "text": "I've seen this movie when I was young, and I remembered it as one of the first films I have truly liked that was not an action movie or a comedy. So, in my later years I decided to watch it again and see if it was just nostalgia or was there really something in that movie. To my surprise, the movie held to my every expectations. It's a great movie. Emotional in the right amount, some jokes, nice songs (not great though, and that actually explains why I did not remember it was a musical) and all in all a great use to my time. I was surprised because the last movies from my childhood that I have revisited did not even pass my minimal demands of a decent movie and yet this movie, which I first saw in the second grade, made me cry today just like it made me cry then. Maybe that's because my dog died recently and maybe not, but the important thing is that it made me feel, and that's why filmmakers make films (that and the money, of course). Yes, there are continuity glitches. Yes, the script has holes, but it doesn't matter. The movie itself is fun and smart. So don't be fooled by cynical people who always look for the bad things in life, because nothing is perfect, and this movie gets a 10 not because it is perfect. It gets 10 simply because it made me feel.",
      "label": "positive",
      "similarity": 0.1300920844078064
    },
    {
      "text": "I really wanted to like this movie, but it was just imposable. The acting was ultra hammy, the plot was annoying, and the pace was SLOW, sooo slowwwwww. The whole time sitting in the theater i wanted the movie to end. Twenty minuets into a films and I'm praying for an ending. Sure some of the visuals were nice, but c'mon guys, I mean really! And for a movie about a guy tuning magical instruments there really wasn't much music to speak of. The music there was was annoying, and boring. There were sound loud shrill sounds at times too, those were also annoying. Mainly this film managed to bore me, and creep me out at the same time.<br /><br />I'm glad its over. I need to go see \"Tideland\" and wash this bad taste out of my mouth.",
      "label": "negative",
      "similarity": 0.1183549165725708
    }
  ]
}
```