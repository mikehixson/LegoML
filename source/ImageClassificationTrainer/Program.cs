using Microsoft.ML;
using Microsoft.ML.Vision;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;

namespace Trainer
{
    class Program
    {
        //const string ImageDataDirectory = @"C:\Users\Mike\Downloads\lego-brick-images\LEGO brick images v1";
        const string ImageDataDirectory = @"C:\Lego";
        const string TrainedModelFilePath = @"C:\Users\Mike\Downloads\ImageClassifier.zip";

        private static MLContext _mlContext;

        static void Main(string[] args)
        {
            RunTrain();
            //RunEvaluate();
        }

        static void RunTrain()
        {
            _mlContext = new MLContext(seed: 1);

            var dataset = LoadData(ImageDataDirectory);
            
            // 4. Split the data 80:20 into train and test sets, train and evaluate.
            var split = _mlContext.Data.TrainTestSplit(dataset, testFraction: 0.2);

            var trainedModel = Train(split.TrainSet, split.TestSet);

            Evaluate(trainedModel, split.TestSet);
        }

        static void RunEvaluate()
        {
            _mlContext = new MLContext(seed: 1);

            var trainedModel = _mlContext.Model.Load(TrainedModelFilePath, out var modelInputSchema);

            var dataset = LoadData(ImageDataDirectory);
            
            // 4. Split the data 80:20 into train and test sets, train and evaluate.
            var split = _mlContext.Data.TrainTestSplit(dataset, testFraction: 0.2);            

            Evaluate(trainedModel, split.TestSet);
        }


        private static IDataView LoadData(string directoryPath)
        {
            var extensions = new[] { ".png", ".jpg" };

            // 2. Load the initial full image-set into an IDataView and shuffle so it'll be better balanced
            var imageDataRecords = Directory.EnumerateFiles(directoryPath, "*.*", SearchOption.AllDirectories)
                .Where(f => extensions.Contains(Path.GetExtension(f)))
                .Select(f => new ImageData(f, Directory.GetParent(f).Name));

            var dataset = _mlContext.Data.LoadFromEnumerable(imageDataRecords);
            dataset = _mlContext.Data.ShuffleRows(dataset);

            // 3. Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical) 
            var estimator = new Microsoft.ML.Data.EstimatorChain<ITransformer>()
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue))
                .Append(_mlContext.Transforms.LoadRawImageBytes(outputColumnName: "Image", inputColumnName: "ImagePath", imageFolder: null));

            var transformer = estimator.Fit(dataset);

            return transformer.Transform(dataset);            
        }

        private static ITransformer Train(IDataView trainDataset, IDataView testDataSet)
        {

            // 5. Define the model's training pipeline using DNN default values
            //var pipeline = _mlContext.MulticlassClassification.Trainers.ImageClassification(featureColumnName: "Image", labelColumnName: "LabelAsKey", validationSet: testDataView)
            //    .Append(_mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));


            // 5.1 (OPTIONAL) Define the model's training pipeline by using explicit hyper-parameters            
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                Epoch = 500,       //200
                BatchSize = 16,
                LearningRate = 0.01f,
                //MetricsCallback = (metrics) => Console.WriteLine(metrics),
                ValidationSet = testDataSet
            };

            var estimator = new Microsoft.ML.Data.EstimatorChain<ITransformer>()
                .Append(_mlContext.MulticlassClassification.Trainers.ImageClassification(options))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));


            Console.WriteLine("*** Training the image classification model ***");
            var watch = Stopwatch.StartNew();

            //Train
            var trainedModel = estimator.Fit(trainDataset);
            
            watch.Stop();
            Console.WriteLine($"Training with transfer learning took: {watch.Elapsed}.");


            // 8. Save the model to assets/outputs (You get ML.NET .zip model file and TensorFlow .pb model file)
            _mlContext.Model.Save(trainedModel, trainDataset.Schema, TrainedModelFilePath);

            return trainedModel;
        }

        private static void Evaluate(ITransformer trainedModel, IDataView testDataSet)
        {
            var predictionsDataView = trainedModel.Transform(testDataSet);

            Console.WriteLine("Making predictions in bulk for evaluating model's quality...");
            var watch = Stopwatch.StartNew();

            var metrics = _mlContext.MulticlassClassification.Evaluate(predictionsDataView, labelColumnName: "LabelAsKey", predictedLabelColumnName: "PredictedLabel");

            watch.Stop();
            Console.WriteLine($"Predicting and Evaluation took: {watch.Elapsed}.");

            ConsoleHelper.PrintMultiClassClassificationMetrics("TensorFlow DNN Transfer Learning", metrics);
        }
    }
}
