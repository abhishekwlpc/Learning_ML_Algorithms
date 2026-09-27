package LINEAR_REGRESSION;

public class LinearRegressionAlgorithm {

    public void train(double[] x, double[] y) {

        double w = 1;
        double b = 1;
        double learning_rate = 0.001;
        int epochs = 1000;

        for (int e = 0; e < epochs; e++) {

            double sumOfErrors = 0;
            double sumOfErrorsX = 0;
            double sumOfSquardErrors =0;
            for (int i = 0; i < x.length; i++) {
                double predictionsOfY = w * x[i] + b;
                double errors = predictionsOfY - y[i];
                sumOfErrors += errors;
                sumOfErrorsX += errors * x[i];
                sumOfSquardErrors += errors * errors;
            }

            double mse = sumOfSquardErrors / x.length;

            double wGradient = ((double) 2 / x.length) * sumOfErrorsX;
            double bGradient = ((double) 2 / x.length) * sumOfErrors;

            w -= (learning_rate * wGradient);
            System.out.println("W : " + w);
            b -= (learning_rate * bGradient);
            System.out.println("B : " + b);

            System.out.println("Mean Squared Error: " + mse);
            System.out.println("Updated weights and bias:");
            System.out.println("w: " + w);
            System.out.println("b: " + b);

        }

    }

}
