package LINEAR_REGRESSION;

public class Main {

    public static void main(String[] args) {

        LinearRegressionAlgorithm linearRegression = new LinearRegressionAlgorithm();

        linearRegression.train(new double[] { 1, 2, 3, 4, 5 }, new double[] { 2, 4, 6, 8, 10 });

    }
}