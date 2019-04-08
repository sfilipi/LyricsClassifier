open System
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Trainers
open System.IO
open FSharp.Data

[<CLIMutable>]
type LyricsInput = 
    {
        Index : int
        Song : string
        Year : string
        Artist : string
        Genre : string
        Lyrics : string
    }

[<CLIMutable>]
type GenrePrediction = 
    {
        [<ColumnName("PredictedLabel")>]
        Genre : string
        Score : float32 []
    }

let downcastPipeline (x : IEstimator<_>) = 
    match x with 
    | :? IEstimator<ITransformer> as y -> y
    | _ -> failwith "downcastPipeline: expecting a IEstimator<ITransformer>"
    
let buildAndTrainTheModel dataSetLocation modelPath =
    
    // Create MLContext to be shared across the model creation workflow objects 
    // Set a random seed for repeatable/deterministic results across multiple trainings.
    let mlContext = MLContext(seed = Nullable 0)

    // STEP 1: Common data loading configuration
    let trainingDataView = mlContext.Data.LoadFromTextFile(dataSetLocation,
                            columns = 
                                [|
                                    TextLoader.Column("Index" , DataKind.String, 0)
                                    TextLoader.Column("Song" , DataKind.String, 1)
                                    TextLoader.Column("Year" , DataKind.String, 2)
                                    TextLoader.Column("Artist" , DataKind.String, 3)
                                    TextLoader.Column("Genre", DataKind.String, 4)
                                    TextLoader.Column("Lyrics", DataKind.String, 5)
                                |], 
                            separatorChar = '\t')
    
    
    let dataProcessPipeline = 
        EstimatorChain()
            .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "Genre"))
            .Append(mlContext.Transforms.Text.FeaturizeText("SongFeaturized", "Song"))
            .Append(mlContext.Transforms.Text.FeaturizeText("ArtistFeaturized", "Artist"))
            .Append(mlContext.Transforms.Concatenate("Features", "SongFeaturized","ArtistFeaturized"))
            .AppendCacheCheckpoint(mlContext)
        |> downcastPipeline
        
    Common.ConsoleHelper.peekDataViewInConsole<LyricsInput> mlContext trainingDataView dataProcessPipeline 2 |> ignore

    // STEP 3: Create the selected training algorithm/trainer
    let trainer =
         mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(
                DefaultColumnNames.Label, 
                DefaultColumnNames.Features)
            |> downcastPipeline

     //Set the trainer/algorithm
    let modelBuilder = 
        dataProcessPipeline
            .Append(trainer)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"))

    //Measure cross-validation time
    let watchCrossValTime = System.Diagnostics.Stopwatch.StartNew()

    let crossValidationResults = 
        mlContext.MulticlassClassification.CrossValidate(data = trainingDataView, estimator = downcastPipeline modelBuilder, numFolds = 6, labelColumn = DefaultColumnNames.Label)

    crossValidationResults
        |> Array.map (fun x -> x.Metrics, x.Model, x.ScoredHoldOutSet) //convert struct tuple for print function
        |> Common.ConsoleHelper.printMulticlassClassificationFoldsAverageMetrics (trainer.ToString())

       
    // STEP 4: Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
    // in order to evaluate and get the model's accuracy metrics
    printfn "=============== Cross-validating to get model's accuracy metrics ==============="

    //Measure cross-validation time
    let watchCrossValTime = System.Diagnostics.Stopwatch.StartNew()

    let crossValidationResults = 
        mlContext.MulticlassClassification.CrossValidate(data = trainingDataView, estimator = downcastPipeline modelBuilder, numFolds = 6, labelColumn = DefaultColumnNames.Label)
        
     //Stop measuring time
    watchCrossValTime.Stop()
    printfn "Time Cross-Validating: %d miliSecs"  watchCrossValTime.ElapsedMilliseconds
           
    crossValidationResults
    |> Array.map (fun x -> x.Metrics, x.Model, x.ScoredHoldOutSet) //convert struct tuple for print function
    |> Common.ConsoleHelper.printMulticlassClassificationFoldsAverageMetrics (trainer.ToString())

    // STEP 5: Train the model fitting to the DataSet
    printfn "=============== Training the model ==============="
    let trainedModel = modelBuilder.Fit(trainingDataView)
    
    let dataSample = { 
       Index = 1
       Year = ""
       Genre = ""
       Artist = ""
       Lyrics = ""
       Song = "BE HUMBLE."
     }
    
    let predEngine = trainedModel.CreatePredictionEngine<LyricsInput, GenrePrediction>(mlContext)
    let prediction =  predEngine.Predict(dataSample)

    printfn "=============== Single Prediction just-trained-model - Result: %s ===============" prediction.Genre

    // STEP 6: Save/persist the trained model to a .ZIP file
    printfn "=============== Saving the model to a file ==============="
    do 
        use f = File.Open(modelPath,FileMode.Create)
        mlContext.Model.Save(trainedModel, f)   



[<EntryPoint>]
let main _argv =
    let appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs().[0])
    let dataDirectoryPath = Path.Combine(appPath,"../../../","Data")
    let dataModelPath = Path.Combine(appPath,"../../../","Data", "Model")
    let trainDataPath  = Path.Combine(appPath,"../../../","Data","lyrics-shorter.csv")


    buildAndTrainTheModel trainDataPath dataModelPath
    
    Console.ReadLine() |> ignore
    0
    