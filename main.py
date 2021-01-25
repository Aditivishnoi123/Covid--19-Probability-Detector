from flask import Flask,render_template,request
import pickle
app=Flask(__name__)
file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()
@app.route('/', methods=["GET","POST"])
def buttonfunction():
    if request.method == "POST":
        myDict=request.form
        fever=int(myDict['FEVER'])
        Bodypain=int(myDict['BODYPAIN'])
        Runnynose=int(myDict['RUNNYNOSE'])
        Diffi=int(myDict['DIFFIBREATHING'])
        throat=int(myDict['Throat Pain'])
        cough=int(myDict['dry cough'])
        taste=int(myDict['loss of taste'])
        smell=int(myDict['loss of smell'])
        age=int(myDict['Age'])
        outside=int(myDict['Outside Visit'])
        
        
        inputfeatures=[fever,Bodypain,Runnynose,Diffi,age,outside,throat,cough,taste,smell]
        INFECTIONPROB=clf.predict_proba([inputfeatures])[0][1]
        print(INFECTIONPROB)
        return render_template('show.html',inf=round(INFECTIONPROB*100))
    
    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)
