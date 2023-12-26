from django.shortcuts import render
import pandas as pd
from .apps import MyappConfig

# Create your views here.
def home(request):
    context = { "say": "Hello world"}
    if request.method == "POST": 
        data = request.POST.dict()
        del data["csrfmiddlewaretoken"]
        df_pred=pd.json_normalize(data)

        X_test = df_pred[[
            "CreditScore",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "Complain",
            "Satisfaction Score",
            "Point Earned",
            "EstimatedSalary",
            "Card Type",
            "Tenure"
        ]].values


        model = MyappConfig.model
        if model.predict(X_test)[0] == 1:
            context = { "result": "ce client est fidéle"}
        else:
            context = { "result" : "ce client n'est pas fidéle"} 


    return render(request, 'home.html', context)
