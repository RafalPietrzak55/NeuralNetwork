using System;

namespace XOR_NeuralNetwork
{
    class Program
    {
        static double[][] inputs = new double[][]
        {
            new double[] {0, 0},
            new double[] {0, 1},
            new double[] {1, 0},
            new double[] {1, 1}
        };

        static double[] targets = new double[] { 0, 1, 1, 0 };

        static double Beta = 1.0;
        static Random rand = new Random();

        static double[,] wInputHidden = new double[2, 2];
        static double[] bHidden = new double[2];
        static double[] wHiddenOutput = new double[2];
        static double bOutput;

        static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-Beta * x));
        }

        static double SigmoidDerivative(double y)
        {
            return Beta * y * (1 - y);
        }

        static void InitializeWeights()
        {
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    wInputHidden[i, j] = rand.NextDouble() * 10 - 5;
            for (int j = 0; j < 2; j++)
                bHidden[j] = rand.NextDouble() * 10 - 5;
            for (int j = 0; j < 2; j++)
                wHiddenOutput[j] = rand.NextDouble() * 10 - 5;
            bOutput = rand.NextDouble() * 10 - 5;
        }

        static double Forward(double[] input)
        {
            double[] hidden = new double[2];
            for (int j = 0; j < 2; j++)
            {
                double sum = bHidden[j];
                for (int i = 0; i < 2; i++)
                    sum += input[i] * wInputHidden[i, j];
                hidden[j] = Sigmoid(sum);
            }

            double sumOut = bOutput;
            for (int j = 0; j < 2; j++)
                sumOut += hidden[j] * wHiddenOutput[j];
            return Sigmoid(sumOut);
        }

        static void Main(string[] args)
        {
            InitializeWeights();
            Console.WriteLine("Test forward pass:");
            for (int i = 0; i < inputs.Length; i++)
            {
                double output = Forward(inputs[i]);
                Console.WriteLine($"Wejście: [{inputs[i][0]}, {inputs[i][1]}], Wyjście sieci: {output:F3}");
            }
        }
    }
}