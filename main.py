from helper_functions import *

while True:
    user_query=input("Enter the question : ")
    if user_query=="exit":
        exit()
    check=(check_greets(user_query))
    
    if check !=True:
        print(check)

    string_value=preprocess(user_query) 

    value=extract_named_entities(string_value)
    
    print(value)



    