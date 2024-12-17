# CS109a-Final-Project

<div>    
    <h5>
        The goal of this project is to train and evaluate predictive models that can classify headline sentiment (positive, neutral, or negative) and detect clickbait using engineered features. Our dataset consists of headlines with corresponding sentiment scores, where the target variable, GPT_score, was categorized into three classes: negative (≤ -0.2), neutral (between -0.2 and 0.4), and positive (≥ 0.4).
    </h5>
    <h5>
        To prepare the data for model training, we implemented feature engineering based on sentiment word analysis. Three lists of sentiment words (negative, neutral, and positive) were manually defined. Additionally, we used the NLTK library to dynamically extract sentiment word lists to enhance our word classification approach. From these lists, we developed several features: flags indicating whether a headline contains negative, neutral, or positive words, counts of such words in the headline, and the density of sentiment words (ratios of sentiment word counts to total word counts).
    </h5>
    <h5>
       We trained multiple models to predict sentiment, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor. For both Random Forest and Gradient Boosting, we fine-tuned the models using GridSearchCV to optimize hyperparameters. For Random Forest, the hyperparameters tuned were n_estimators and max_depth, while for Gradient Boosting, we tuned n_estimators, learning_rate, and max_depth. 
    </h5>
     <h5>
        Model performance was evaluated using overall accuracy, class-specific accuracy, and mean squared error (MSE) where applicable. For Linear Regression, the baseline model, the overall training accuracy was 55.86%, while the test accuracy was 53.82%. However, a deeper look into the class-specific performance revealed significant imbalances. The model achieved excellent accuracy for the neutral class (~96% on both training and test sets), but performed poorly for the negative class (~25%) and the positive class (~1.7%). This result indicates that the dataset is heavily skewed toward the neutral class, making it challenging for the model to predict minority classes effectively.
    </h5>
    <h5>
        Several challenges were encountered during the project. The most prominent issue was class imbalance, where neutral sentiment dominated the dataset, leading the model to prioritize predicting neutral headlines. To address this, future improvements include implementing oversampling techniques like SMOTE or assigning class weights to the model during training. Additionally, while our manually defined sentiment word lists provided a starting point, the model’s performance suggests that the features could be further improved. Techniques like word embeddings (Word2Vec or BERT), TF-IDF scores, and syntactic features may better capture the sentiment nuances in headlines.
    </h5>
    <h5>
        Linear models like regression assume a linear relationship between features and the target variable, which is likely insufficient for sentiment classification. As a result, using non-linear models such as Random Forest and Gradient Boosting offers greater promise for capturing complex patterns in the data. Evaluation metrics like F1-Score and confusion matrices could further provide deeper insights into the model’s performance across all sentiment classes.
    </h5>
    <h5>
        In conclusion, this project outlines a systematic approach to feature engineering, model training, and evaluation for predicting headline sentiment and clickbait. While the initial models performed ok for the dominant neutral class, significant improvements are required for predicting positive and negative sentiments. Future work will focus on addressing class imbalance, improving feature representation, and exploring advanced machine learning models to enhance overall performance.
    </h5>
</div>
