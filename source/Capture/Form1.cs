using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace Capture
{
    public partial class Form1 : Form
    {
        // Create class-level accesible variables
        //VideoCapture capture;
        //Mat frame;
        //Bitmap image;
        //private Thread camera;
        //bool isCameraRunning = false;
        CancellationTokenSource cancelTokenSource = new CancellationTokenSource();




        private async void CaptureCamera2(CancellationToken cancellationToken)
        {
            var progress = new Progress<(Image Image, string Prediction)>(p =>
            {
                if (pictureBox1.Image != null)
                    pictureBox1.Image.Dispose();

                pictureBox1.Image = p.Image;

                if (p.Prediction != null)
                    textBox1.Text = p.Prediction;
            });

            await Task.Factory.StartNew(() => CaptureCameraCallback2(progress, cancellationToken));
        }

        private void CaptureCameraCallback2(IProgress<(Image Image, string Prediction)> progress, CancellationToken cancellationToken)
        {
            var url = "http://192.168.1.13:8080/video";


            //capture = new VideoCapture(0);
            //capture.Open(0);
            using (var capture = new VideoCapture(url))
            using (var frame = new Mat())
            {
                capture.Open(url);

                long frameNumber = 0;

                if (capture.IsOpened())
                {
                    while (!cancellationToken.IsCancellationRequested)
                    {

                        capture.Read(frame);
                        var image = BitmapConverter.ToBitmap(frame);

                        string prediction = null;

                        if (frameNumber % 10 == 0)
                            prediction = Predicter.Go(frame.ToBytes());

                        progress.Report((image, prediction));

                        frameNumber++;
                    }
                }

                capture.Release();
            }
        }




        public Form1()
        {
            InitializeComponent();
        }

        // When the user clicks on the start/stop button, start or release the camera and setup flags
        private async void button1_Click(object sender, EventArgs e)
        {
            if (button1.Text.Equals("Start"))
            {
                button1.Text = "Stop";

                cancelTokenSource = new CancellationTokenSource();

                CaptureCamera2(cancelTokenSource.Token);
            }
            else
            {
                button1.Text = "Start";

                cancelTokenSource.Cancel();
            }
        }

        // When the user clicks on take snapshot, the image that is displayed in the pictureBox will be saved in your computer
        private void button2_Click(object sender, EventArgs e)
        {
            if (pictureBox1.Image != null)
            {
                // Take snapshot of the current image generate by OpenCV in the Picture Box
                Bitmap snapshot = new Bitmap(pictureBox1.Image);

                // Save in some directory
                // in this example, we'll generate a random filename e.g 47059681-95ed-4e95-9b50-320092a3d652.png
                // snapshot.Save(@"C:\Users\sdkca\Desktop\mysnapshot.png", ImageFormat.Png);
                snapshot.Save(string.Format(@"C:\Users\Mike\Downloads\Test\{0}.png", Guid.NewGuid()), ImageFormat.Png);
            }
            else
            {
                Console.WriteLine("Cannot take picture if the camera isn't capturing image!");
            }
        }
    }
}
