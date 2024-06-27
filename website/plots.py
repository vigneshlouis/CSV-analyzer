from flask import Blueprint,render_template,flash,redirect
import io


from flask import request

import base64

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import numpy as np
hist= Blueprint('hist',__name__)


@hist.route('/gettingfile',methods=['POST','GET'])
def gettfile():
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


                plttype = "plot"
                flash("uploaded successfully",category="success")
            return render_template("home.html",insights1=True,heads1=columns,heads=heads,non_num1=non_num,plt_type=plttype,gf='gf',gt=gt,nrow=nrow,ncol1=ncol,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value)
    except Exception as e:
        flash(f"{e}",category='error')
        return redirect("/")







@hist.route('/lineplot',methods=['POST','GET'])
def lineplot():
    try:
        plttype = 'lineplot'
        upload_path = ['website/uploads', f.filename ]
        df= pd.read_csv('/'.join(upload_path))

        heads = df.columns.tolist()
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])
        columns = numeric.columns.tolist()

        nrow, ncol = df.shape
        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']

        non_num = []
        for col in heads:
            if col not in columns:
                non_num.append(col)

        selected = request.form.getlist('check_box')
        if selected:
            selectnumeric = []
            for cols in selected:
                if cols in columns:
                    selectnumeric.append(cols)

            x = selectnumeric[0]
            y = selectnumeric[1]

        else:
            DF = pd.DataFrame(df)
            numeric = DF.select_dtypes(include=['int64', 'float64'])
            columns = numeric.columns.tolist()
            x = columns[0]
            y = columns[1]

        plt.figure(figsize=(8, 8))

        sns.lineplot(data=DF, x=f"{x}", y=f"{y}", marker='o', linestyle='-')  # 'o' for markers, '-' for lines
        plt.title(f'Line Plot of {x} vs. {y}')
        plt.xlabel(f"{x}")
        plt.ylabel(f"{y}")
        img=io.BytesIO()
        plt.savefig(img,format='png')
        img.seek(0)
        plot_data=base64.b64encode(img.read()).decode()

        return render_template('home.html',insights1=True, plot_data=plot_data,nrow=nrow,ncol1=ncol,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value,heads=heads,plttype=plttype,headss=heads,num=columns,non_num=non_num)
    except:
        flash("dataset not supported for this plot,try other plots for better viisualization",category='error')
        return  redirect('/')



@hist.route('/histogram',methods=['POST','GET'])
def histogram():
    try:

        plttype = 'histogram'
        upload_path = ['website/uploads', f.filename ]
        df = pd.read_csv('/'.join(upload_path))
        heads = df.columns.tolist()
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])
        columns = numeric.columns.tolist()
        nrow, ncol = df.shape
        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']
        non_num = []
        for col in heads:
            if col not in columns:
                non_num.append(col)

        selected = request.form.getlist('check_box')
        if selected:


            x = selected[0]
        else:

            x = heads[0]


        plt.figure(figsize=(8,8))


        plt.hist(df[f"{x}"], bins=20, color='lightseagreen',  edgecolor='black',alpha=0.7)

        plt.xlabel(f"{x}")
        plt.ylabel('Frequency')
        plt.title("histogram")
        plt.xticks(rotation=90)
        plt.legend()
        img=io.BytesIO()
        plt.savefig(img,format='png')
        img.seek(0)
        plot_data=base64.b64encode(img.read()).decode()
        return render_template('home.html', plot_data=plot_data,insights1=True,nrow=nrow,ncol1=ncol,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value,heads=heads,plttype=plttype,headss=heads,plt_type=plttype,num=columns,non_num=non_num,pltt_type='histogram')

    except:
        flash("file not submitted or please select single column ,try again",category='error')
        return  redirect('/')


@hist.route('/heatmap',methods=["POST","GET"])
def heatmap():
    try:



        plttype = 'heatmap'
        upload_path = ['website/uploads', f.filename, ]
        df = pd.read_csv('/'.join(upload_path))
        plt.figure(figsize=(8,8))
        heads = df.columns.tolist()

        nrow, ncol = df.shape
        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])

        sns.heatmap(numeric.corr(),annot=True,cmap='coolwarm')
        plt.title("correlation")
        plt.legend()



        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()
        return render_template('home.html', plot_data=plot_data,insights1=True,nrow=nrow,ncol1=ncol,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value,heads=heads,plttype=plttype,plt_type=plttype,headss=heads)

    except:
        flash("file not submitted,try again", category='error')
        return redirect('/')

@hist.route('/boxplot',methods=['POST','GET'])
def boxplot():
    try:

        plttype = 'boxplot'
        upload_path = ['website/uploads', f.filename, ]
        df = pd.read_csv('/'.join(upload_path))
        plt.figure(figsize=(8,8))
        heads = df.columns.tolist()
        nrow, ncol = df.shape
        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])

        plt.boxplot(numeric.corr())
        plt.title("correlation")
        plt.xlabel("Columns")
        plt.ylabel("correlation coefficient")
        '''plt.xticks(range(1,len(heads)+1),heads,rotation=90)'''
        plt.legend()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()
        return render_template('home.html', plot_data=plot_data,insights1=True,nrow=nrow,ncol1=ncol,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value,heads=heads,plttype=plttype,headss=heads,plt_type=plttype)

    except:
        flash("file not submitted,try again", category='error')
        return redirect('/')

@hist.route('/scatterplot',methods=["POST",'GET'])
def scatterplot():
    try:
        plttype = 'scatterplot'
        upload_path = ['website/uploads', f.filename, ]
        df = pd.read_csv('/'.join(upload_path))
        heads = df.columns.tolist()
        nrow, ncol = df.shape
        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])
        columns = numeric.columns.tolist()
        non_num = []
        for col in heads:
            if col not in columns:
                non_num.append(col)

        selected = request.form.getlist('check_box')
        if selected:
            selectnumeric = []
            for cols in selected:
                if cols in columns:
                    selectnumeric.append(cols)

            x = selectnumeric[0]
            y = selectnumeric[1]

        else:
            DF = pd.DataFrame(df)
            numeric = DF.select_dtypes(include=['int64', 'float64'])
            columns = numeric.columns.tolist()
            x = columns[0]
            y = columns[1]

        plt.figure(figsize=(8,8))

        plt.scatter(df[f"{x}"], df[f"{y}"])
        plt.legend()

        plt.title("scatterplot")
        plt.xlabel(f"{x}")
        plt.ylabel(f"{y}")
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()
        return render_template('home.html', plot_data=plot_data,insights1=True,nrow=nrow,ncol1=ncol,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value,heads=heads,plttype=plttype,headss=heads,plt_type=plttype,num=columns,non_num=non_num)

    except:
        flash("file not submitted,try again", category='error')
        return redirect('/')


@hist.route('/barplot',methods=["POST","GET"])
def barplot():
    try:
        plttype = 'barplot'
        upload_path = ['website/uploads', f.filename, ]
        df = pd.read_csv('/'.join(upload_path))
        heads = df.columns.tolist()
        nrow, ncol = df.shape
        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])
        columns = numeric.columns.tolist()
        non_num=[]
        for col in heads:
            if col not in columns:
                non_num.append(col)

        selected = request.form.getlist('check_box')
        if selected:
            selectnumeric = []
            for cols in selected:
                if cols in columns:
                    selectnumeric.append(cols)

            x = selectnumeric[0]
            y = selectnumeric[1]

        else:
            DF = pd.DataFrame(df)
            numeric = DF.select_dtypes(include=['int64', 'float64'])
            columns = numeric.columns.tolist()
            x = columns[0]
            y = columns[1]

        plt.figure(figsize=(8,8))

        plt.bar(df[f'{x}'], df[f"{y}"])
        plt.title("bar plot")
        plt.xlabel(f"{x}")
        plt.ylabel(f"{y}")

        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()
        return render_template('home.html', plot_data=plot_data,insights1=True,nrow=nrow,ncol1=ncol,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value,heads=heads,plttype=plttype,headss=heads,plt_type=plttype,num=columns,non_num=non_num,pltt_type='barplot')

    except:
        flash("file not submitted,try again", category='error')
        return redirect('/')


@hist.route('/pairplot', methods=["POST"])
def pairplot():

    try:


        plttype = 'pairplot'
        upload_path = ['website/uploads', f.filename, ]
        df = pd.read_csv('/'.join(upload_path))
        heads = df.columns.tolist()
        nrow, ncol = df.shape
        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']
        plt.figure(figsize=(8, 8))
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])

        sns.pairplot(numeric,diag_kind='kde')
        plt.title("pair plot")

        plt.legend()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()
        return render_template('home.html', plot_data=plot_data,insights1=True,nrow=nrow,ncol1=ncol,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value, heads=heads,plttype=plttype,headss=heads)

    except:
        flash("file not submitted,try again", category='error')
        return redirect('/')


@hist.route('/piechart', methods=["POST"])
def piechart():
    try:
        plttype = 'piechart'
        upload_path = ['website/uploads', f.filename, ]
        df = pd.read_csv('/'.join(upload_path))
        heads = df.columns.tolist()
        nrow, ncol = df.shape
        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']

        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])
        columns = numeric.columns.tolist()
        non_num = []
        for col in heads:
            if col not in columns:
                non_num.append(col)

        selected = request.form.getlist('check_box')
        print(selected)
        if selected:
            selectnumeric = []
            for cols in selected:
                if cols in columns:
                    selectnumeric.append(cols)

            x = selectnumeric[0]


        else:
            DF = pd.DataFrame(df)
            numeric = DF.select_dtypes(include=['int64', 'float64'])
            columns = numeric.columns.tolist()
            x = columns[0]

        colors = ['#ff9999','#D0E7D2','#C08261', '#66b3ff', '#99ff99', '#ffcc99','#001524','#C70039','#E9B824','#5B0888','#618264','#9D76C1']
        plt.figure(figsize=(8, 13))

        category_counts = df[x].value_counts()
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)



        plt.title(f"{x}-piechart")

        plt.legend(title=x, loc="lower right")

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()
        return render_template('home.html', plot_data=plot_data,insights1=True,nrow=nrow,ncol1=ncol,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value, heads=heads,plttype=plttype,headss=heads,num=columns,non_num=non_num,pltt_type='piechart')

    except:
        flash("file not submitted,try again", category='error')
        return redirect('/')

@hist.route('/regrplot',methods=['POST','GET'])
def regressionplot():
    try:
        plttype = 'regrplot'
        upload_path = ['website/uploads', f.filename, ]
        df = pd.read_csv('/'.join(upload_path))
        heads = df.columns.tolist()
        nrow, ncol = df.shape
        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])
        columns = numeric.columns.tolist()
        non_num = []
        for col in heads:
            if col not in columns:
                non_num.append(col)

        selected = request.form.getlist('check_box')
        if selected :
            selectnumeric=[]
            for cols in selected:
                if cols in columns:
                    selectnumeric.append(cols)


            x = selectnumeric[0]
            y = selectnumeric[1]

        else:
            DF = pd.DataFrame(df)
            numeric = DF.select_dtypes(include=['int64', 'float64'])
            columns = numeric.columns.tolist()
            x = columns[0]
            y = columns[1]

        plt.figure(figsize=(8, 8))
        sns.regplot(data=df,x=f"{x}",y=f"{y}")
        plt.title("regression plot")
        plt.legend()



        img=io.BytesIO()
        plt.savefig(img,format='png')
        img.seek(0)
        plot_data=base64.b64encode(img.read()).decode()

        return render_template('home.html', plot_data=plot_data,insights1=True,nrow=nrow,ncol1=ncol,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value,heads=heads,plttype=plttype,headss=heads,num=columns,non_num=non_num)

    except:
        flash("file not submitted,try again",category='error')
        return  redirect('/')


@hist.route('/violinplot',methods=['POST','GET'])
def violinplt():
    try:
        plttype = 'violinplot'
        upload_path = ['website/uploads', f.filename, ]
        df = pd.read_csv('/'.join(upload_path))
        heads = df.columns.tolist()
        nrow, ncol = df.shape
        summary_stats = df.describe()

        count = summary_stats.loc['count']
        mean = summary_stats.loc['mean']
        std = summary_stats.loc['std']
        min_value = summary_stats.loc['min']
        percentile_25 = summary_stats.loc['25%']
        median = summary_stats.loc['50%']
        percentile_75 = summary_stats.loc['75%']
        max_value = summary_stats.loc['max']
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])
        columns = numeric.columns.tolist()
        non_num = []
        for col in heads:
            if col not in columns:
                non_num.append(col)

        selected = request.form.getlist('check_box')
        if selected :
            selectnumeric=[]
            for cols in selected:
                if cols in columns:
                    selectnumeric.append(cols)


            x = selectnumeric[0]
            y = selectnumeric[1]

        else:
            DF = pd.DataFrame(df)
            numeric = DF.select_dtypes(include=['int64', 'float64'])
            columns = numeric.columns.tolist()
            x = columns[0]
            y = columns[1]

        sns.set_theme(style="ticks", palette="pastel")

        plt.figure(figsize=(8, 8))
        sns.violinplot(x=f'{x}', y=f'{y}', data=df)
        plt.title("violin plot")
        plt.legend()



        img=io.BytesIO()
        plt.savefig(img,format='png')
        img.seek(0)
        plot_data=base64.b64encode(img.read()).decode()

        return render_template('home.html', plot_data=plot_data,insights1=True,nrow=nrow,ncol1=ncol,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value,heads=heads,plttype=plttype,headss=heads,num=columns,non_num=non_num)

    except:
        flash("file not submitted,try again",category='error')
        return  redirect('/')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@hist.route('/line3d',methods=['POST','GET'])
def line3d():

    fig = plt.figure(figsize=(8, 8))
    plttype = 'line3d'

    upload_path = ['website/uploads', f.filename, ]
    df = pd.read_csv('/'.join(upload_path))
    heads = df.columns.tolist()
    nrow, ncol = df.shape
    summary_stats = df.describe()

    count = summary_stats.loc['count']
    mean = summary_stats.loc['mean']
    std = summary_stats.loc['std']
    min_value = summary_stats.loc['min']
    percentile_25 = summary_stats.loc['25%']
    median = summary_stats.loc['50%']
    percentile_75 = summary_stats.loc['75%']
    max_value = summary_stats.loc['max']
    DF = pd.DataFrame(df)
    numeric = DF.select_dtypes(include=['int64', 'float64'])
    columns = numeric.columns.tolist()
    non_num = []
    for col in heads:
        if col not in columns:
            non_num.append(col)

    selected = request.form.getlist('check_box')
    if selected :
        selectnumeric=[]
        for cols in selected:
            if cols in columns:
                selectnumeric.append(cols)


        x = selectnumeric[0]
        y = selectnumeric[1]

    else:
        DF = pd.DataFrame(df)
        numeric = DF.select_dtypes(include=['int64', 'float64'])

        columns = numeric.columns.tolist()
        x = columns[0]
        y = columns[1]
    print(DF[f'{x}'].shape,DF[f'{y}'].shape)
    shapee=DF[f'{x}'].shape[0]
    print(shapee)
    z=np.linspace(0, 5, shapee)


    ax = fig.add_subplot(111, projection='3d')

    # Create the 3D line plot
    ax.plot(DF[x].to_numpy(),DF[y].to_numpy(), z, label='3D Line')


    img=io.BytesIO()
    fig.savefig(img,format='png')
    img.seek(0)
    plot_data=base64.b64encode(img.read()).decode()

    return render_template('home.html', plot_data=plot_data,insights1=True,nrow=nrow,ncol1=ncol,count=count,mean=mean,std=std,min_value=min_value,percentile_75=percentile_75,percentile_25=percentile_25,median=median,max_value=max_value,heads=heads,plttype=plttype,headss=heads,num=columns,non_num=non_num)

#except:
#flash("file not submitted,try again",category='error')
#return  redirect('/')



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming you have your dataset loaded into a DataFrame df
# If not, replace this with your dataset loading code

from flask import request  # Assuming you are using Flask

# Define the route for your function
@hist.route('/surface3d', methods=['POST', 'GET'])
def surface3d():
    fig = plt.figure(figsize=(10, 10))
    plttype = 'surface3d'

    upload_path = ['website/uploads', f.filename, ]
    df = pd.read_csv('/'.join(upload_path))
    heads = df.columns.tolist()
    nrow, ncol = df.shape
    summary_stats = df.describe()

    count = summary_stats.loc['count']
    mean = summary_stats.loc['mean']
    std = summary_stats.loc['std']
    min_value = summary_stats.loc['min']
    percentile_25 = summary_stats.loc['25%']
    median = summary_stats.loc['50%']
    percentile_75 = summary_stats.loc['75%']
    max_value = summary_stats.loc['max']
    DF = pd.DataFrame(df)
    numeric = DF.select_dtypes(include=['int64', 'float64'])
    columns = numeric.columns.tolist()
    non_num = []
    for col in heads:
        if col not in columns:
            non_num.append(col)

    # Get the user's selection for the three columns
    selected = request.form.getlist('check_box')
    if len(selected) >= 3:
        x, y, z = selected[:3]
    else:
        # If the user didn't select three columns, provide default columns
        x, y, z = columns[:3]

    # Extract the selected columns as numpy arrays
    # Extract the selected columns as numpy arrays
    # Extract the selected columns as numpy arrays
    X = df[x].values
    Y = df[y].values
    Z = df[z].values

    # Reshape Z to match the shape of X and Y
    if len(X) == len(Y) == len(Z):
        shapee = X.shape[0], Y.shape[0]

        X, Y = np.meshgrid(X, Y)

        # Reshape the Z values to match the shape of X and Y
        Z = Z.reshape((X.shape[0],1))
        print(Z)

        ax = fig.add_subplot(111, projection='3d')

        # Create the 3D surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False, rstride=1, cstride=1)

        # Add labels
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.set_title(f'3D Surface Plot for {x}, {y}, and {z}')
        cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
        cbar.set_label('Z-values', rotation=270)

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()

        return render_template('home.html', plot_data=plot_data, nrow=nrow, ncol1=ncol, count=count, mean=mean,
                               std=std, min_value=min_value, percentile_75=percentile_75, percentile_25=percentile_25,
                               median=median, max_value=max_value, heads=heads, plttype=plttype, headss=heads,
                               num=columns,insights1=True, non_num=non_num)


    else:
        # Handle the case where the columns have different lengths
        # You can provide an error message or handle it as needed
        print("Columns have different lengths.")
        flash("Columns have different lengths.", category='error')
        return redirect('/')

    # Create the 3D surface plot

@hist.route('/bubble3d', methods=['POST', 'GET'])
def bubble3d():
    fig = plt.figure(figsize=(10, 10))
    plttype = 'bubble3d'

    upload_path = ['website/uploads', f.filename, ]
    df = pd.read_csv('/'.join(upload_path))
    heads = df.columns.tolist()
    nrow, ncol = df.shape
    summary_stats = df.describe()

    count = summary_stats.loc['count']
    mean = summary_stats.loc['mean']
    std = summary_stats.loc['std']
    min_value = summary_stats.loc['min']
    percentile_25 = summary_stats.loc['25%']
    median = summary_stats.loc['50%']
    percentile_75 = summary_stats.loc['75%']
    max_value = summary_stats.loc['max']
    DF = pd.DataFrame(df)
    numeric = DF.select_dtypes(include=['int64', 'float64'])
    columns = numeric.columns.tolist()
    non_num = []
    for col in heads:
        if col not in columns:
            non_num.append(col)

    selected = request.form.getlist('check_box')
    if len(selected) >= 3:
        x, y, z = selected[:3]
    else:
        x, y, z = columns[:3]

    X = df[x].values
    Y = df[y].values
    Z = df[z].values

    # Ensure that X, Y, and Z have the same length
    if len(X) == len(Y) == len(Z):
        sizes = (df[f'{z}'] + 1) * 30  # Adjust the size calculation based on your specific requirement

        ax = fig.add_subplot(111, projection='3d')

        # Create the bubble plot
        ax.scatter(X, Y, Z, c='b', marker='o', s=sizes, alpha=0.7, cmap='viridis', label='Bubble Plot')

        # Set labels
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        cbar = plt.colorbar(ax.scatter(X, Y, Z, c=sizes, cmap='viridis'))
        cbar.set_label('Size')
        ax.set_title('Bubble 3D Plot', fontsize=16)


        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()

        return render_template('home.html', plot_data=plot_data, nrow=nrow, ncol1=ncol, count=count, mean=mean,
                               std=std, min_value=min_value, percentile_75=percentile_75, percentile_25=percentile_25,
                               median=median, max_value=max_value, heads=heads, plttype=plttype, headss=heads,
                               num=columns,insights1=True, non_num=non_num)

    else:
        # Handle the case where the columns have different lengths
        print("Columns have different lengths.")
        flash("Columns have different lengths.", category='error')
        return redirect('/')




@hist.route('/scatter3d', methods=['POST', 'GET'])
def scatter3d():
    fig = plt.figure(figsize=(10, 10))
    plttype = 'scatter3d'

    upload_path = ['website/uploads', f.filename, ]
    df = pd.read_csv('/'.join(upload_path))
    heads = df.columns.tolist()
    nrow, ncol = df.shape
    summary_stats = df.describe()

    count = summary_stats.loc['count']
    mean = summary_stats.loc['mean']
    std = summary_stats.loc['std']
    min_value = summary_stats.loc['min']
    percentile_25 = summary_stats.loc['25%']
    median = summary_stats.loc['50%']
    percentile_75 = summary_stats.loc['75%']
    max_value = summary_stats.loc['max']
    DF = pd.DataFrame(df)
    numeric = DF.select_dtypes(include=['int64', 'float64'])
    columns = numeric.columns.tolist()
    non_num = []
    for col in heads:
        if col not in columns:
            non_num.append(col)

    selected = request.form.getlist('check_box')
    if len(selected) >= 3:
        x, y, z = selected[:3]
    else:
        x, y, z = columns[:3]

    X = df[x].values
    Y = df[y].values
    Z = df[z].values

    # Ensure that X, Y, and Z have the same length
    if len(X) == len(Y) == len(Z):
        ax = fig.add_subplot(111, projection='3d')

        # Create the 3D scatter plot
        ax.scatter(X, Y, Z, c='b', marker='o', cmap='viridis', label='3D Scatter Plot')

        # Set labels
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.set_title('3D Scatter Plot', fontsize=16)

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()

        return render_template('home.html', plot_data=plot_data, nrow=nrow, ncol1=ncol, count=count, mean=mean,
                               std=std, min_value=min_value, percentile_75=percentile_75, percentile_25=percentile_25,
                               median=median, max_value=max_value, heads=heads, plttype=plttype, headss=heads,
                               num=columns,insights1=True, non_num=non_num)

    else:
        # Handle the case where the columns have different lengths
        print("Columns have different lengths.")
        flash("Columns have different lengths.", category='error')
        return redirect('/')



@hist.route('/contour3d', methods=['POST', 'GET'])
def contour3d():
    fig = plt.figure(figsize=(10, 10))
    plttype = 'contour3d'

    upload_path = ['website/uploads', f.filename, ]
    df = pd.read_csv('/'.join(upload_path))
    heads = df.columns.tolist()
    nrow, ncol = df.shape
    summary_stats = df.describe()

    count = summary_stats.loc['count']
    mean = summary_stats.loc['mean']
    std = summary_stats.loc['std']
    min_value = summary_stats.loc['min']
    percentile_25 = summary_stats.loc['25%']
    median = summary_stats.loc['50%']
    percentile_75 = summary_stats.loc['75%']
    max_value = summary_stats.loc['max']
    DF = pd.DataFrame(df)
    numeric = DF.select_dtypes(include=['int64', 'float64'])
    columns = numeric.columns.tolist()
    non_num = []
    for col in heads:
        if col not in columns:
            non_num.append(col)

    selected = request.form.getlist('check_box')
    if len(selected) >= 3:
        x, y, z = selected[:3]
    else:
        x, y, z = columns[:3]

    X = df[x].values
    Y = df[y].values
    Z = df[z].values

    # Ensure that X, Y, and Z have the same length
    if len(X) == len(Y) == len(Z):
        ax = fig.add_subplot(111, projection='3d')

        # Create a meshgrid for the contour plot
        # Create a meshgrid for the contour plot
        x_values = np.linspace(min(X), max(X), ncol)
        y_values = np.linspace(min(Y), max(Y), nrow)
        X_mesh, Y_mesh = np.meshgrid(x_values, y_values)
        Z_mesh = np.interp(np.ravel(X_mesh), X, Z)

        # Reshape Z_mesh to be 2D
        Z_mesh = Z_mesh.reshape(X_mesh.shape)

        # Create the 3D contour plot
        ax.contour3D(X_mesh, Y_mesh, Z_mesh, 50, cmap='viridis')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.set_title('3D Contour Plot', fontsize=16)

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()

        return render_template('home.html', plot_data=plot_data, nrow=nrow, ncol1=ncol, count=count, mean=mean,
                               std=std, min_value=min_value, percentile_75=percentile_75, percentile_25=percentile_25,
                               median=median, max_value=max_value, heads=heads, plttype=plttype, headss=heads,
                               num=columns,insights1=True, non_num=non_num)

    else:
        # Handle the case where the columns have different lengths
        print("Columns have different lengths.")
        flash("Columns have different lengths.", category='error')
        return redirect('/')

@hist.route('/ribbon3d', methods=['POST', 'GET'])
def ribbon3d():
    fig = plt.figure(figsize=(10, 10))
    plttype = 'ribbon3d'

    upload_path = ['website/uploads', f.filename, ]
    df = pd.read_csv('/'.join(upload_path))
    heads = df.columns.tolist()
    nrow, ncol = df.shape
    summary_stats = df.describe()

    count = summary_stats.loc['count']
    mean = summary_stats.loc['mean']
    std = summary_stats.loc['std']
    min_value = summary_stats.loc['min']
    percentile_25 = summary_stats.loc['25%']
    median = summary_stats.loc['50%']
    percentile_75 = summary_stats.loc['75%']
    max_value = summary_stats.loc['max']
    DF = pd.DataFrame(df)
    numeric = DF.select_dtypes(include=['int64', 'float64'])
    columns = numeric.columns.tolist()
    non_num = []
    for col in heads:
        if col not in columns:
            non_num.append(col)

    selected = request.form.getlist('check_box')
    if len(selected) >= 3:
        x, y, z = selected[:3]
    else:
        x, y, z = columns[:3]

    X = df[x].values
    Y = df[y].values
    Z = df[z].values

    # Ensure that X, Y, and Z have the same length
    if len(X) == len(Y) == len(Z):
        ax = fig.add_subplot(111, projection='3d')

        # Create a 3D ribbon plot by drawing lines
        cmap = plt.get_cmap('viridis')
        for i in range(len(X) - 1):
            ax.plot([X[i], X[i + 1]], [Y[i], Y[i + 1]], [Z[i], Z[i + 1]],  color=cmap(0.5),linestyle='--', linewidth=5)

        # Set labels
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.set_title('3D Ribbon Plot', fontsize=16)

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()

        return render_template('home.html', plot_data=plot_data, nrow=nrow, ncol1=ncol, count=count, mean=mean,
                               std=std, min_value=min_value, percentile_75=percentile_75, percentile_25=percentile_25,
                               median=median, max_value=max_value, heads=heads, plttype=plttype, headss=heads,
                               num=columns, insights1=True,non_num=non_num)

    else:
        # Handle the case where the columns have different lengths
        print("Columns have different lengths.")
        flash("Columns have different lengths.", category='error')
        return redirect('/')



@hist.route('/bar3d', methods=['POST', 'GET'])
def bar3d():
    fig = plt.figure(figsize=(10, 10))
    plttype = 'bar3d'

    upload_path = ['website/uploads', f.filename, ]
    df = pd.read_csv('/'.join(upload_path))
    heads = df.columns.tolist()
    nrow, ncol = df.shape
    summary_stats = df.describe()

    count = summary_stats.loc['count']
    mean = summary_stats.loc['mean']
    std = summary_stats.loc['std']
    min_value = summary_stats.loc['min']
    percentile_25 = summary_stats.loc['25%']
    median = summary_stats.loc['50%']
    percentile_75 = summary_stats.loc['75%']
    max_value = summary_stats.loc['max']
    DF = pd.DataFrame(df)
    numeric = DF.select_dtypes(include=['int64', 'float64'])
    columns = numeric.columns.tolist()
    non_num = []
    for col in heads:
        if col not in columns:
            non_num.append(col)

    selected = request.form.getlist('check_box')
    if len(selected) >= 3:
        x, y, z = selected[:3]
    else:
        x, y, z = columns[:3]

    X = df[x].values
    Y = df[y].values
    Z = df[z].values


    if len(X) == len(Y) == len(Z):




        ax = fig.add_subplot(111, projection='3d')

        # Create a 3D bar plot
        ax.bar3d(X, Y, 0, 1, 1, Z, shade=True,color='red')

        # Set labels
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.set_title('3D Bar Plot', fontsize=16)

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.read()).decode()

        return render_template('home.html', plot_data=plot_data, nrow=nrow, ncol1=ncol, count=count, mean=mean,
                               std=std, min_value=min_value, percentile_75=percentile_75, percentile_25=percentile_25,
                               median=median, max_value=max_value, heads=heads, plttype=plttype, headss=heads,
                               num=columns,insights1=True, non_num=non_num)

    else:
        # Handle the case where the columns have different lengths
        print("Columns have different lengths.")
        flash("Columns have different lengths.", category='error')
        return redirect('/')


