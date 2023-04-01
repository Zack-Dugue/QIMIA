

Experiment takes 3 Arguments in main:
    A path to the model JSON, which will be of this format
        Model_Name : String
        Args : Object
        Kwargs : Object
    A path to the data
        Data_Name : String
        Args : Object
        Kwargs : Object
    A path to the training params of this format:
        Optimizer:
            optimizer_name : String
            args : Object
            kwargs : Object
        LR_Scheduler :
            optimizer_name : String
            args : Array
            kwargs : Object
        Num_Epochs : Int



There will be a parser function, for the model and the args.
Inside of experiment there will be a dictionary, containing a mapping
from the "names" of the models to the models themselves.
The models will then be initialized using Arg and Kwarg.
Same thing for the data.

