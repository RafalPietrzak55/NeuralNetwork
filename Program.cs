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

        static void Main(string[] args)
        {
            Console.WriteLine("Dane XOR:");
            for (int i = 0; i < inputs.Length; i++)
            {
                Console.WriteLine($"Wejście: [{inputs[i][0]}, {inputs[i][1]}], Oczekiwane wyjście: {targets[i]}");
            }
        }
    }
}