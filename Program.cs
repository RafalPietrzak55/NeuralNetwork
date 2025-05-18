using System;

namespace XOR_NeuralNetwork
{
    class Program
    {
        // Parametry sieci
        static int inputSize = 2;
        static int hiddenSize = 2;
        static int outputSize = 1;

        // Dane XOR
        static double[][] inputs = new double[][]
        {
            new double[] {0, 0},
            new double[] {0, 1},
            new double[] {1, 0},
            new double[] {1, 1}
        };

        static double[][] targets = new double[][]
        {
            new double[] {0},
            new double[] {1},
            new double[] {1},
            new double[] {0}
        };

        // Parametry uczenia
        static double Beta = 1.0;
        static double LearningRate = 0.3;
        static int Epochs = 50000;
        static Random rand = new Random();

        // Wagi i biasy
        static double[,] wInputHidden = new double[2, 2];
        static double[] bHidden = new double[2];
        static double[,] wHiddenOutput = new double[2, 1];
        static double[] bOutput = new double[1];

        // Funkcja aktywacji
        static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-Beta * x));
        }

        // Pochodna sigmoidy względem wyjścia
        static double SigmoidDerivative(double y)
        {
            return Beta * y * (1 - y);
        }

        // Losowa inicjalizacja wag i biasów w zakresie [-5, 5]
        static void InitializeWeights()
        {
            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    wInputHidden[i, j] = rand.NextDouble() * 10 - 5;
            for (int j = 0; j < hiddenSize; j++)
                bHidden[j] = rand.NextDouble() * 10 - 5;
            for (int j = 0; j < hiddenSize; j++)
                for (int k = 0; k < outputSize; k++)
                    wHiddenOutput[j, k] = rand.NextDouble() * 10 - 5;
            for (int k = 0; k < outputSize; k++)
                bOutput[k] = rand.NextDouble() * 10 - 5;
        }

        // Propagacja w przód
        static double[] Forward(double[] input)
        {
            double[] hidden = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++)
            {
                double sum = bHidden[j];
                for (int i = 0; i < inputSize; i++)
                    sum += input[i] * wInputHidden[i, j];
                hidden[j] = Sigmoid(sum);
            }

            double[] output = new double[outputSize];
            for (int k = 0; k < outputSize; k++)
            {
                double sum = bOutput[k];
                for (int j = 0; j < hiddenSize; j++)
                    sum += hidden[j] * wHiddenOutput[j, k];
                output[k] = Sigmoid(sum);
            }
            return output;
        }

        // Uczenie sieci na jednej próbce
        static void Train(double[] input, double[] target)
        {
            // Forward pass
            double[] hidden = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++)
            {
                double sum = bHidden[j];
                for (int i = 0; i < inputSize; i++)
                    sum += input[i] * wInputHidden[i, j];
                hidden[j] = Sigmoid(sum);
            }

            double[] output = new double[outputSize];
            for (int k = 0; k < outputSize; k++)
            {
                double sum = bOutput[k];
                for (int j = 0; j < hiddenSize; j++)
                    sum += hidden[j] * wHiddenOutput[j, k];
                output[k] = Sigmoid(sum);
            }

            // Backward pass
            double[] deltaOutput = new double[outputSize];
            for (int k = 0; k < outputSize; k++)
            {
                double error = target[k] - output[k];
                deltaOutput[k] = error * SigmoidDerivative(output[k]);
            }

            double[] deltaHidden = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < outputSize; k++)
                    sum += deltaOutput[k] * wHiddenOutput[j, k];
                deltaHidden[j] = sum * SigmoidDerivative(hidden[j]);
            }

            // Aktualizacja wag i biasów warstwy wyjściowej
            for (int j = 0; j < hiddenSize; j++)
                for (int k = 0; k < outputSize; k++)
                    wHiddenOutput[j, k] += LearningRate * deltaOutput[k] * hidden[j];
            for (int k = 0; k < outputSize; k++)
                bOutput[k] += LearningRate * deltaOutput[k];

            // Aktualizacja wag i biasów warstwy ukrytej
            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    wInputHidden[i, j] += LearningRate * deltaHidden[j] * input[i];
            for (int j = 0; j < hiddenSize; j++)
                bHidden[j] += LearningRate * deltaHidden[j];
        }

        static void Main(string[] args)
        {
            // Inicjalizacja wag
            InitializeWeights();

            // Uczenie przez wiele epok
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                // Mieszanie kolejności próbek (opcjonalnie)
                int[] order = { 0, 1, 2, 3 };
                for (int i = 0; i < order.Length; i++)
                {
                    int j = rand.Next(i, order.Length);
                    int tmp = order[i];
                    order[i] = order[j];
                    order[j] = tmp;
                }
                for (int idx = 0; idx < inputs.Length; idx++)
                {
                    int i = order[idx];
                    Train(inputs[i], targets[i]);
                }
            }

            // Samokontrola: testowanie po nauce
            Console.WriteLine("Testowanie po nauce:");
            for (int i = 0; i < inputs.Length; i++)
            {
                var output = Forward(inputs[i]);
                double error = Math.Abs(targets[i][0] - output[0]);
                Console.WriteLine($"Wejście: [{inputs[i][0]}, {inputs[i][1]}], Wyjście: {output[0]:F3}, Oczekiwane: {targets[i][0]}, Błąd: {error:F3}");
            }
            Console.WriteLine("Naciśnij dowolny klawisz, aby zakończyć...");
            Console.ReadKey();
        }
    }
}