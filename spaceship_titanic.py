import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import numpy as np
import pandas as pd
import xgboost as xgb


def main():
    train_df = pd.read_csv("spaceship-titanic/data/train.csv")
    test_df = pd.read_csv("spaceship-titanic/data/test.csv")

    train_df['dataset'] = "train"
    test_df['dataset'] = "test"

    joined_df = pd.concat([train_df, test_df])

    nan_vals = predict_categorical_features(joined_df)

    for feature, vals in nan_vals.items():
        foo = joined_df[feature].isna()
        joined_df.loc[foo, feature] = vals
    
    Wealthiest_Deck = joined_df.groupby('cabin_deckandside_code').aggregate({'total_spending': 'sum', 'PassengerId': 'size'}).reset_index()
    # Create DeckAverageSpent feature
    Wealthiest_Deck['deckandside_average_spent'] = Wealthiest_Deck['total_spending'] / Wealthiest_Deck['PassengerId']
    
    joined_df = joined_df.merge(Wealthiest_Deck[["cabin_deckandside_code", "deckandside_average_spent"]], how = 'left', on = ['cabin_deckandside_code'])

    train_df = joined_df[joined_df['dataset'] == 'train'].drop(columns=['dataset'])
    test_df = joined_df[joined_df['dataset'] == 'test'].drop(columns=['dataset'])


    # features = ['PassengerId','HomePlanet','CryoSleep','Cabin','Destination','Age','VIP',        'RoomService','FoodCourt','ShoppingMall','Spa','VRDeck',       'Name','Transported']
    # features = ['cabin_deck_code', 'HomePlanet_code', 'VIP_code', 'CryoSleep_code', 'is_alone', 'Destination_code', 'age_cat_code', 'total_spending', 'spending_cat_code']
    features = ['cabin_deckandside_code', 'VIP_code', 'CryoSleep_code', 'HomePlanet_code', 'Destination_code', 'luxury_spending', 'regular_spending', 'num_travelling', 'age_cat_code', 'deckandside_average_spent']
    

    X = train_df[features]
    X_test = test_df[features]
    y = train_df['Transported'].astype(int)

    
    assert X.isna().sum().any() == False
    assert X_test.isna().sum().any() == False
    assert y.isna().sum().any() == False
    # x_train, x_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        # tf.keras.layers.Dropout(0.5),  # Optional dropout layer for regularization
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Optional dropout layer for regularization
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification, so using sigmoid activation
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    paramgrid = {
        'max_depth': list(range(1, 20, 2)),
        'n_estimators': list(range(1, 200, 20))
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=1), paramgrid)

    # Fit the grid search model
    grid_search.fit(X, y)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Create a new RandomForestClassifier instance with the best parameters
    rf = RandomForestClassifier(**best_params, random_state=1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index = features)
    importances.plot(kind = 'barh', figsize = (12, 8))
    plt.show()

    rf_pred = rf.predict(X_test)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

    model.fit(X, y, epochs=15, batch_size=40, callbacks=[early_stopping])

    pred = model.predict(X_test)
    # test_loss, test_acc = model.evaluate(x_test, y_test)
    # print(test_acc)

    # print(f"rf: {accuracy_score(pred, y_test)}")
    # Save the DataFrame to a CSV file
    results = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': np.round(pred).astype(bool).flatten()})
    rf_results = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': np.round(rf_pred).astype(bool).flatten()})
    results.to_csv(f'spaceship-titanic/data/nn_predictions.csv', index=False)
    rf_results.to_csv(f'spaceship-titanic/data/rf_predictions.csv', index=False)


def predict_categorical_features(df):
    
    df = feature_engineer(df)

    
    # encode with nan vals then drop based on uncoded column
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'cabin_deckandside'] #'cabin_deck', 'cabin_side']
    regressor_features = ['total_spending', 'num_travelling']
    
    label_encoders = {}
    for feature in categorical_features:
        encoder = LabelEncoder()
        df[f'{feature}_code'] = encoder.fit_transform(df[feature])
        label_encoders[feature] = encoder

    nan_vals = {}
    
    for feature in categorical_features:
        X = df[[f'{f}_code' for f in categorical_features] + regressor_features + [feature]].dropna()
        X = X.drop([f'{feature}_code', feature], axis=1)

        y = df[[f"{feature}_code", feature]].dropna()
        y = np.ravel(y.drop(feature, axis=1))
        # x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
        
        x_test = df[df[feature].isnull()]
        x_test = x_test.drop(f'{feature}_code', axis=1)

        features = []
        for f in categorical_features:
            if f != feature:
                features.append(f'{f}_code')

        assert x_test[features + regressor_features].isna().sum().any() == False
        assert X.isna().sum().any() == False
        assert len(X) == len(y)


        model = RandomForestClassifier()
        model.fit(X, y)
        pred = model.predict(x_test[features + regressor_features])
        
        encoder = label_encoders[feature]
        temp = pd.DataFrame()
        temp['PassengerId'] = x_test['PassengerId']
        temp[feature] = encoder.inverse_transform(pred)
        nan_vals[feature] = temp
        
    return nan_vals    


def feature_engineer(df):
    
    encoder = LabelEncoder()

    df['id'] = df['PassengerId'].apply(lambda x: x.split('_')[0])

    df['Age'] = df['Age'].fillna(df['Age'].mean().round())
    df["age_cat"] = pd.cut(df['Age'], bins = [-0.1, 12.0, 19.0, 40.0, 60.0, 80.0], labels = ['0 - 12', '13 - 19', '20 - 40', '41 - 61', '61 - 80'])
    df['age_cat_code'] = encoder.fit_transform(df['age_cat'])

    spending_features = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    for feature in spending_features:
        # fill na values with 0 if age < 13 or cryosleep true
        df[feature] = np.where((df["Age"] < 13) | (df["CryoSleep"] == True), 0, df[feature])
        df[feature] = df[feature].fillna(df.groupby("age_cat")[feature].transform('mean'))

    df['total_spending'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']

    df['spending_cat'] = pd.cut(df['total_spending'], bins = [-0.1, 1000.0, 5000.0, 20000.0, 100000.0], labels = ['0 - 1k', '1 - 5k', '5 - 20k', '20k+',])

    df['luxury_spending'] = df['Spa'] + df['VRDeck'] + df['RoomService']
    df['regular_spending'] = df['FoodCourt'] + df['ShoppingMall']

    df['num_travelling'] = df.groupby('id').transform('size') - 1
    df['is_alone'] = df['num_travelling'].apply(lambda x: 0 if x > 0 else 1)


    df['cabin_deck'] = df['Cabin'].apply(lambda x: str(x).split("/")[0] if pd.notna(x) else np.nan)
    df['cabin_no'] = df['Cabin'].apply(lambda x: str(x).split("/")[1] if pd.notna(x) else np.nan)
    df['cabin_side'] = df['Cabin'].apply(lambda x: str(x).split("/")[2] if pd.notna(x) else np.nan)
    df['cabin_deckandside'] = df['cabin_deck'] + df['cabin_side']
    
    return df


if __name__ == "__main__":
    main()