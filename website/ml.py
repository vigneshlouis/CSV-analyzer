from flask import Blueprint,render_template,flash,redirect

from sklearn.metrics import accuracy_score

from flask import request

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

from sklearn.metrics import mean_squared_error, r2_score




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

ml = Blueprint('ml',__name__)

@ml.route('/predict')
def predictpage():
    return render_template("predictpage.html")

@ml.route('/getfile',methods=['POST','GET'])
def getfile():
    try:
        global f
        global heads

        if request.method == 'POST':
            f = request.files['file']
            if f:
                upload_path=['website/uploads',f.filename]
                f.save("/".join(upload_path))

                df = pd.read_csv('/'.join(upload_path))
                heads = df.columns.tolist()



                DF = pd.DataFrame(df)
                numeric = DF.select_dtypes(include=['int64', 'float64'])
                columns = numeric.columns.tolist()
                gt=columns
                non_num = []
                for col in heads:
                    if col not in columns:
                        non_num.append(col)
                nrow,ncol= df.shape
                summary_stats = df.describe()

                count = summary_stats.loc['count']
                mean = summary_stats.loc['mean']
                std = summary_stats.loc['std']
                min_value = summary_stats.loc['min']
                percentile_25 = summary_stats.loc['25%']
                median = summary_stats.loc['50%']
                percentile_75 = summary_stats.loc['75%']
                max_value = summary_stats.loc['max']


                #plttype = "plot"
                flash("uploaded successfully",category="success")
        return render_template("predictpage.html",heads1=columns,heads=heads,num_col=columns,non_num=non_num,gf='gf',gt=gt,nrow=nrow,ncol1=ncol,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value)
    except Exception as e:
        flash(f"file not selected{e}",category='error')
        return redirect("/predict")


@ml.route('/linearreg',methods=['POST','GET'])
def linearreg():
    upload_path = ['website/uploads', f.filename]
    df = pd.read_csv('/'.join(upload_path))

    heads = df.columns.tolist()
    DF = pd.DataFrame(df)
    numeric = DF.select_dtypes(include=['int64', 'float64'])
    columns = numeric.columns.tolist()
    non_num = []
    for col in heads:
        if col not in columns:
            non_num.append(col)

    summary_stats = df.describe()

    count = summary_stats.loc['count']
    mean = summary_stats.loc['mean']
    std = summary_stats.loc['std']
    min_value = summary_stats.loc['min']
    percentile_25 = summary_stats.loc['25%']
    median = summary_stats.loc['50%']
    percentile_75 = summary_stats.loc['75%']
    max_value = summary_stats.loc['max']

    models= 'linear'
    return render_template("predictpage.html", models=models, heads1=columns, heads=heads, num_col=columns,non_num=non_num, count=count, mean=mean, std=std, min_value=min_value,percentile_75=percentile_75, percentile_25=percentile_25, median=median, max_value=max_value)







@ml.route('/linearop',methods=["POST",'GET'])
def linearop():
    upload_path = ['website/uploads', f.filename]
    df = pd.read_csv('/'.join(upload_path))

    heads = df.columns.tolist()
    DF = pd.DataFrame(df)
    numeric = DF.select_dtypes(include=['int64', 'float64'])
    columns = numeric.columns.tolist()
    non_num = []
    for col in heads:
        if col not in columns:
            non_num.append(col)

    summary_stats = df.describe()

    count = summary_stats.loc['count']
    mean = summary_stats.loc['mean']
    std = summary_stats.loc['std']
    min_value = summary_stats.loc['min']
    percentile_25 = summary_stats.loc['25%']
    median = summary_stats.loc['50%']
    percentile_75 = summary_stats.loc['75%']
    max_value = summary_stats.loc['max']

    target = request.form.get('target')

    input1 = request.form.get('input1')
    print(target)
    print(input1)
    categorical_columns = df.select_dtypes(include=['object']).columns
    print(categorical_columns, 'fufufty')
    categor = df.drop(columns=target, axis=1)
    categor1 = categor.select_dtypes(include=['object']).columns
    print(categor1, 'catgor')

    if not categorical_columns.empty:
        labelencoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = labelencoder.fit_transform(df[col])

    enn = labelencoder.classes_
    print(enn)
    nulll = df.isnull().sum()
    if not nulll.empty:

        missing_columns = df.columns[df.isna().any()].tolist()
        for col in missing_columns:
            df[col].fillna(df[col].mean(), inplace=True)
    print(df.head())
    x = df.drop(columns=target, axis=1)
    print(x.head())

    cola = x.columns.tolist()
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

    model = LinearRegression()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_train)



    num_array1 = input1.split(',')
    num_array = np.asarray(num_array1)
    input_data = np.where(num_array == '', 1, num_array)

    print(input_data)

    input_data = input_data.reshape(1, -1)
    dff = pd.DataFrame(input_data, columns=cola)
    print(dff.head())

    if not categor1.empty:
        labelencoder = LabelEncoder()
        for col in categor1:
            dff[col] = labelencoder.fit_transform(dff[col])
    print(dff.head())
    enn = labelencoder.classes_
    en = labelencoder.classes_
    print(en)

    nulllo = dff.isnull().sum()
    print(nulllo)
    if not nulllo.empty:

        miss_columns = dff.columns[dff.isna().any()].tolist()
        for none in miss_columns:
            dff[none].fillna(dff[none].mean(), inplace=True)
    dff.fillna(value=0, axis=1)

    print(dff.head())

    prediction = model.predict(dff)
    print(prediction)
    n = len(enn)
    index = [i for i in range(n)]
    if prediction in index:
        prediction = enn[prediction]

    models = 'linear'
    '''plt.figure(figsize=(8, 7))
    sns.regplot(x=prediction,y=y_pred)
    plt.title("pair plot")
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = base64.b64encode(img.read()).decode()'''



    flash("scroll down to see result",category='success')
    return render_template("predictpage.html", prediction=prediction, heads1=columns, heads=heads, num_col=columns, non_num=non_num,count=count, mean=mean, std=std, min_value=min_value, percentile_75=percentile_75,percentile_25=percentile_25, median=median, max_value=max_value, models=models)





@ml.route('/logisticregui',methods=['POST','GET'])
def logisticreg():
    try:
        upload_path = ['website/uploads', f.filename]
        df = pd.read_csv('/'.join(upload_path))

        heads = df.columns.tolist()
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])
        columns = numeric.columns.tolist()
        non_num = []
        for col in heads:
            if col not in columns:
                non_num.append(col)



        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']
        models='logic'

        return render_template("predictpage.html",models=models,heads1=columns,heads=heads,num_col=columns,non_num=non_num,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value)
    except:
        return redirect('/predict')

@ml.route("/logisticregop",methods=['POST','GET'])
def logisticoutput():
    try:
        upload_path = ['website/uploads', f.filename]
        df = pd.read_csv('/'.join(upload_path))


        heads = df.columns.tolist()
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])
        columns = numeric.columns.tolist()
        non_num = []
        for col in heads:
            if col not in columns:
                non_num.append(col)


        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']

        target = request.form.get('target')

        input1 = request.form.get('input1')
        print(target)
        print(input1)
        categorical_columns = df.select_dtypes(include=['object']).columns
        print(categorical_columns,'fufufty')
        categor=df.drop(columns=target,axis=1)
        categor1=categor.select_dtypes(include=['object']).columns
        print(categor1,'catgor')


        if not categorical_columns.empty:
            labelencoder = LabelEncoder()
            for col in categorical_columns:
                df[col] = labelencoder.fit_transform(df[col])

        enn=labelencoder.classes_
        print(enn)
        nulll=df.isnull().sum()
        if not nulll.empty:

            missing_columns = df.columns[df.isna().any()].tolist()
            for col in missing_columns:
                df[col].fillna(df[col].mean(), inplace=True)
        print(df.head())
        x = df.drop(columns=target,axis=1)
        print(x.head())

        cola=x.columns.tolist()
        y = df[target]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

        model = LogisticRegression()

        model.fit(x_train, y_train)

        x_train_pred = model.predict(x_train)
        train_accuracy = accuracy_score( y_train,x_train_pred)
        x_test_pred = model.predict(x_test)
        test_accuracy = accuracy_score( y_test,x_test_pred)

        num_array1 = input1.split(',')
        num_array = np.asarray(num_array1)
        input_data=np.where(num_array == '',1,num_array)


        print(input_data)

        input_data = input_data.reshape(1, -1)
        dff=pd.DataFrame(input_data,columns=cola)
        print(dff.head())

        if not categor1.empty:
            labelencoder = LabelEncoder()
            for col in categor1:
                dff[col] = labelencoder.fit_transform(dff[col])
        print(dff.head())
        en=labelencoder.classes_
        print(en)

        '''nulllo = dff.isnull().sum()
        print(nulllo)
        if not nulllo.empty:
    
            miss_columns = dff.columns[dff.isna().any()].tolist()
            for none in miss_columns:
                dff[none].fillna(dff[none].mean(), inplace=True)'''
        dff.fillna(value=0, axis=1)

        print(dff.head())

        prediction = model.predict(dff)
        print(prediction)
        n=len(enn)
        index=[i for i in range(n)]
        if prediction in index:
            prediction=enn[prediction]


        models='logic'


        flash("scroll down to see result", category='success')
        return render_template("predictpage.html",models=models,train_accuracy=train_accuracy,test_accuracy=test_accuracy, prediction=prediction, heads1=columns, heads=heads, num_col=columns, non_num=non_num,count=count, mean=mean, std=std, min_value=min_value, percentile_75=percentile_75,percentile_25=percentile_25, median=median, max_value=max_value)
    except Exception as e:
        flash(f"{e}", category='error')
        return redirect('/getfile')



import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


@ml.route('/decisionui',methods=['POST','GET'])
def decisionstart():
    try:
        upload_path = ['website/uploads', f.filename]
        df = pd.read_csv('/'.join(upload_path))

        heads = df.columns.tolist()
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])
        columns = numeric.columns.tolist()
        non_num = []
        for col in heads:
            if col not in columns:
                non_num.append(col)



        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']
        models='decision'

        return render_template("predictpage.html",models=models,heads1=columns,heads=heads,num_col=columns,non_num=non_num,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value)
    except AttributeError:
        return redirect('/predict')

@ml.route('/decisiontree',methods=["POST",'GET'])
def deciontree():
    upload_path = ['website/uploads', f.filename]
    df = pd.read_csv('/'.join(upload_path))

    heads = df.columns.tolist()
    DF = pd.DataFrame(df)
    numeric = DF.select_dtypes(include=['int64', 'float64'])
    columns = numeric.columns.tolist()
    non_num = []
    for col in heads:
        if col not in columns:
            non_num.append(col)

    summary_stats = df.describe()

    count = summary_stats.loc['count']
    mean = summary_stats.loc['mean']
    std = summary_stats.loc['std']
    min_value = summary_stats.loc['min']
    percentile_25 = summary_stats.loc['25%']
    median = summary_stats.loc['50%']
    percentile_75 = summary_stats.loc['75%']
    max_value = summary_stats.loc['max']

    target = request.form.get('target')

    input1 = request.form.get('input1')
    print(target)
    print(input1)
    categorical_columns = df.select_dtypes(include=['object']).columns
    print(categorical_columns, 'fufufty')
    categor = df.drop(columns=target, axis=1)
    categor1 = categor.select_dtypes(include=['object']).columns
    print(categor1, 'catgor')

    if not categorical_columns.empty:
        labelencoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = labelencoder.fit_transform(df[col])

    enn = labelencoder.classes_
    print(enn)
    nulll = df.isnull().sum()
    if not nulll.empty:

        missing_columns = df.columns[df.isna().any()].tolist()
        for col in missing_columns:
            df[col].fillna(df[col].mean(), inplace=True)
    print(df.head())
    x = df.drop(columns=target, axis=1)
    print(x.head())

    cola = x.columns.tolist()
    y = df[target]





    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(x, y)

    # Two  lines to make our compiler able to draw:
    classes=df[target].unique()
    plt.figure(figsize=(12, 8))  # You can adjust the figure size as needed
    tree.plot_tree(dtree, filled=True, feature_names=x.columns.tolist(),class_names=classes.astype(str).tolist())
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = base64.b64encode(img.read()).decode()


    flash("scroll down to see result",category='success')
    return render_template("predictpage.html",models='decision',plot_data=plot_data, heads1=columns, heads=heads, num_col=columns, non_num=non_num,count=count, mean=mean, std=std, min_value=min_value, percentile_75=percentile_75,percentile_25=percentile_25, median=median, max_value=max_value)

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
@ml.route("/svm",methods=["POST","GET"])
def svmui():
    try:
        upload_path = ['website/uploads', f.filename]
        df = pd.read_csv('/'.join(upload_path))

        heads = df.columns.tolist()
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])
        columns = numeric.columns.tolist()
        non_num = []
        for col in heads:
            if col not in columns:
                non_num.append(col)

        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']
        models = 'svm'

        return render_template("predictpage.html", models=models, heads1=columns, heads=heads, num_col=columns,non_num=non_num, count=count, mean=mean, std=std, min_value=min_value,percentile_75=percentile_75, percentile_25=percentile_25, median=median,max_value=max_value)
    except AttributeError:
        return redirect('/predict')

@ml.route("/svmop",methods=["POST","GET"])
def svmop():




    # Function to load and preprocess the CSV dataset
    def load_and_preprocess_csv(file_path):
        # Load the CSV file into a DataFrame
        upload_path = ['website/uploads', f.filename]
        df = pd.read_csv('/'.join(upload_path))

        # Check if the dataset has any missing values and handle them if needed
        if df.isnull().values.any():
            df.dropna(inplace=True)

        # Check if the dataset contains categorical features and encode them if needed
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            label_encoders = {}
            for col in categorical_columns:
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col])

        return df

    # Function to train an SVM classifier and plot the decision boundary
    def train_and_plot_svm(df, target_column):
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Create and train the SVM classifier
        clf = svm.SVC(kernel='linear', C=1.0)
        clf.fit(X_train, y_train)

        # Display confusion matrix
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Calculate and display accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    if __name__ == "__main__":
        # Specify the path to your CSV file
        csv_file_path = r"C:\Users\ELCOT\PycharmProjects\CSV Analyzer\website\uploads\diabetes.csv"

        # Specify the name of the target column in your CSV dataset
        target_column_name = request.form.get("target")

        # Load and preprocess the CSV dataset
        df = load_and_preprocess_csv(csv_file_path)

        # Train SVM and display evaluation metrics
        train_and_plot_svm(df, target_column_name)



    flash("scroll down to see result", category='success')
    return "<p>hell0<p>"

@ml.route("/randomforestui",methods=["POST","GET"])
def rf():
    try:
        upload_path = ['website/uploads', f.filename]
        df = pd.read_csv('/'.join(upload_path))

        heads = df.columns.tolist()
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])
        columns = numeric.columns.tolist()
        non_num = []
        for col in heads:
            if col not in columns:
                non_num.append(col)

        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']
        models = 'rf'

        return render_template("predictpage.html", models=models, heads1=columns, heads=heads, num_col=columns,
                               non_num=non_num, count=count, mean=mean, std=std, min_value=min_value,
                               percentile_75=percentile_75, percentile_25=percentile_25, median=median,
                               max_value=max_value)
    except:
        return redirect('/predict')
@ml.route("/rfop",methods=["POST","GET"])
def rfop():
    try:
        upload_path = ['website/uploads', f.filename]
        df = pd.read_csv('/'.join(upload_path))

        heads = df.columns.tolist()
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])
        columns1 = numeric.columns.tolist()
        non_num = []
        for col in heads:
            if col not in columns1:
                non_num.append(col)

        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']

        target = request.form.get('target')

        input1 = request.form.get('input1')
        print(target)
        print(input1)
        categorical_columns = df.select_dtypes(include=['object']).columns
        print(categorical_columns, 'fufufty')
        categor = df.drop(columns=target, axis=1)
        categor1 = categor.select_dtypes(include=['object']).columns
        print(categor1, 'catgor')

        if not categorical_columns.empty:
            labelencoder = LabelEncoder()
            for col in categorical_columns:
                df[col] = labelencoder.fit_transform(df[col])

        enn = labelencoder.classes_
        print(enn)
        nulll = df.isnull().sum()
        if not nulll.empty:

            missing_columns = df.columns[df.isna().any()].tolist()
            for col in missing_columns:
                df[col].fillna(df[col].mean(), inplace=True)
        print(df.head())
        X = df.drop(columns=target, axis=1)
        print(X.head())

        cola = X.columns.tolist()
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        accuracy =accuracy * 100

        # Plot feature importances
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        num_array1 = input1.split(',')
        num_array = np.asarray(num_array1)
        input_data = np.where(num_array == '', 1, num_array)

        print(input_data)

        input_data = input_data.reshape(1, -1)
        dff = pd.DataFrame(input_data, columns=cola)
        print(dff.head())

        if not categor1.empty:
            labelencoder = LabelEncoder()
            for col in categor1:
                dff[col] = labelencoder.fit_transform(dff[col])
        print(dff.head())
        en = labelencoder.classes_
        print(en)


        dff.fillna(value=0, axis=1)

        print(dff.head())

        prediction = clf.predict(dff)
        print(prediction)
        n = len(enn)
        index = [i for i in range(n)]
        if prediction in index:
            prediction = enn[prediction]

        plt.figure(figsize=(8, 8))
        plt.title("Random Forest Feature Importances")
        plt.bar(range(X_train.shape[1]), importances[indices], align="center")
        plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()
        models = 'rf'

        flash("scroll down to see result", category='success')
        return render_template("predictpage.html",plot_data=plot_data, models=models,
                               accuracy=accuracy, prediction=prediction, heads1=columns1, heads=heads,
                               num_col=columns1, non_num=non_num, count=count, mean=mean, std=std, min_value=min_value,
                               percentile_75=percentile_75, percentile_25=percentile_25, median=median, max_value=max_value)

    except:
        flash("submit your file and try again", category='error')
        return redirect('/getfile')



