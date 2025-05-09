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

        static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-Beta * x));
        }

        static double SigmoidDerivative(double y)
        {
            return Beta * y * (1 - y);
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Test funkcji sigmoidalnej:");
            for (double x = -2; x <= 2; x += 1)
            {
                double y = Sigmoid(x);
                Console.WriteLine($"x={x}, sigmoid(x)={y:F3}, sigmoid'(x)={SigmoidDerivative(y):F3}");
            }
        }
    }
}