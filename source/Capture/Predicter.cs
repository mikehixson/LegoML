using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Capture
{
    public static class Predicter
    {
        const string TrainedModelFilePath = @"C:\Users\Mike\Downloads\ImageClassifier.zip";

        private static readonly PredictionEngine<InMemoryImageData, ImagePrediction> _predictionEngine;

        static Predicter()
        {
            var mlContext = new MLContext(seed: 1);

            // Load the model
            var loadedModel = mlContext.Model.Load(TrainedModelFilePath, out var modelInputSchema);

            // Create prediction engine to try a single prediction (input = InMemoryImageData, output = ImagePrediction)
            _predictionEngine = mlContext.Model.CreatePredictionEngine<InMemoryImageData, ImagePrediction>(loadedModel);
        }

        public static string Go(byte[] image)
        {
            var imageToPredict = new InMemoryImageData(image, String.Empty, String.Empty);

            var prediction = _predictionEngine.Predict(imageToPredict);

            return $"Predicted Label : [{prediction.PredictedLabel}], " +
                  $"Probability : [{prediction.Score.Max()}]";
        }


        private class InMemoryImageData
        {
            public readonly byte[] Image;
            public readonly string Label;
            public readonly string ImageFileName;

            public InMemoryImageData(byte[] image, string label, string imageFileName)
            {
                Image = image;
                Label = label;
                ImageFileName = imageFileName;
            }
        }

        private class ImagePrediction
        {
            [ColumnName("Score")]
            public float[] Score;

            [ColumnName("PredictedLabel")]
            public string PredictedLabel;
        }
    }
}
