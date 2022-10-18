# I301D-Assignment2
In this project, I analyzed a dataset of Wikipedia comments made available by Jigsaw, a subsidiary of Google that created the Perspective tool. This dataset includes a unique comment id, the text of the comment, and a series of binary labels applied by human raters: "toxic," "severe_toxic," "obscene," "threat," "insult," and "identity_hate" + an appended "score" column, which represents the toxicity score assigned to the comment text by the live version of the Perspective API. The data is available under a CC0 license.

My hypothesis is that the Perspective API will make more mistakes in classifying comments as toxic or not toxic if they contain more Internet slang acronyms, specifically 'lol', 'lmao', 'lmfao', 'wth','wtf', 'jk', 'idk', 'smh', 'ikr', and 'tbh'. I am interested in seeing how well Perspective API will perform with these non-conventional words as they have become well-integrated in dialogue from younger generations, but may not be understood by the API.

# Analysis
I first tested out the Perspective API's scoring of several different phrases that I would have considered as positive, negative, neutral, or explicit. I wasn't suprised to see that explicit words scored very high while positive phrases like "I love you" scored low. What surprised me were some more ambiguous words such as "ugly", "fat' or "hate" that scored varyingly (0.636, 0.447, 0.271). This affected the set toxicity threshold I chose (0.4) and piqued my curiosity as to how well Perspective API would perform with less conventional words and with slang words instead. 

I decided to check how well the Perspective API performed in general with the whole dataset, based off the threshold I set. From this, I found that the Perspective API labels much more comments as toxic than not. The Perspective API had a greater ratio of true positives to false positives (240.715), compared to true negatives to false negatives(0.862). Essentially, it seemed like that the Perspective API did a worse job in labeling toxic content (88.188% accuracy) than non toxic content (96.525%). From this, I was interested to see how it would do with Internet slang content.

Notedly, I was working with a small sample (n=145) of content with the selected Internet slang acronyms, something that could impact the accuracy of my results and making it more difficult in being able to compared with the complete dataset (n=41338). However, I would assume that because this was coming from Wikipedia comments as opposed to a social media platform, where more Internet slang and more users that would be more liekly to use the slang would be found.

With this sample, I found that the Perspective API marked almost evenly half of the content as toxic versus non toxic, possibly speaking to how Internet slang can be used evenly in both toxic and non toxic contexts. Even more interestingly, I found that the Perspective API did a worse job in labeling toxic content (69.811% accuracy) than non toxic content (100%). Compared with how it did with marking the general dataset (88.188% accuracy marking true toxic content), it did worse. I believe the accuracy to marking toxic content as toxic comes from other context that other words in the content could have provided, such as curse words that may have accompanied the slang. On the other hand, I think that the Perspective API struggles with content that has Internet slang with not much other context from more conventional words.

Overall though, this sample proves my hypothesis to be true as the Perspective API performed worse with comments that had Internet slang acronynms. This test points to something important to consider when teaching API how to perform such a task. As the social content on the Internet becomes more casual and more slang arises, I believe that the relevance and efficiency of Perspective API should be considered in this social context. The Perspective API may not be able to provide helpful insights if it cannot adapt to the growing culture of Internet slang.
