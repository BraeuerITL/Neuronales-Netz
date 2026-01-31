package neuronalesnetz;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class App {

    private int MAXINPUTS = 25;

    /*
    von: https://www.computerwoche.de/a/so-bauen-sie-ein-neuronales-netzwerk-auf,3613718
     */
    public static void main(String[] args) {
        new App().trainAndPredict();

    }

    public void trainAndPredict() {

        /*
        TRAININGSDATEN
         */
        List<List<Double>> data = new ArrayList<>();
        data.add(Arrays.asList(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0));
        data.add(Arrays.asList(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0));
        data.add(Arrays.asList(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0));
        data.add(Arrays.asList(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0));
        /*
        KORREKTE ANTWORTEN
         */
        List<Double> answers = Arrays.asList(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        /*
        Zwei Netzwerke erstellen und trainieren lassen.
         */
        Network network1 = new Network(10000, 3.0);
        network1.train(data, answers);

        Network network2 = new Network(30000, 3.0);
        network2.train(data, answers);

        /*
        Netzwerke testen mit Beispieldaten.
         */
        System.out.println("Vorhersagen:");
        System.out.println("Lernfaktoren: network" + network1.epochs + "=" + network1.learnFactor + " | network" + network2.epochs + "=" + network2.learnFactor);
        List<Double> input1 = Arrays.asList(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        List<Double> input2 = Arrays.asList(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        List<Double> input3 = Arrays.asList(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        List<Double> input4 = Arrays.asList(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
        List<Double> input5 = Arrays.asList(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
        List<Double> input6 = Arrays.asList(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);

        System.out.println(String.format(" falsch: network" + network1.epochs + ": %.10f | network" + network2.epochs + ": %.10f", network1.predict(input1), network2.predict(input1)));
        System.out.println(String.format("richtig: network" + network1.epochs + ": %.10f | network" + network2.epochs + ": %.10f", network1.predict(input2), network2.predict(input2)));
        System.out.println(String.format("richtig: network" + network1.epochs + ": %.10f | network" + network2.epochs + ": %.10f", network1.predict(input3), network2.predict(input3)));
        System.out.println(String.format(" falsch: network" + network1.epochs + ": %.10f | network" + network2.epochs + ": %.10f", network1.predict(input4), network2.predict(input4)));
        System.out.println(String.format("richtig: network" + network1.epochs + ": %.10f | network" + network2.epochs + ": %.10f", network1.predict(input5), network2.predict(input5)));
        System.out.println(String.format("richtig: network" + network1.epochs + ": %.10f | network" + network2.epochs + ": %.10f", network1.predict(input6), network2.predict(input6)));

    }

    class Network {

        int epochs = 0;
        Double learnFactor = null;
        List<Neuron> neurons = Arrays.asList(
                new Neuron(), new Neuron(), new Neuron(),
                new Neuron(), new Neuron(),
                new Neuron());

        public Network(int epochs) {
            this.epochs = epochs;
        }

        public Network(int epochs, Double learnFactor) {
            this.epochs = epochs;
            this.learnFactor = learnFactor;
        }

        /*
        Obwohl die Liste der Neuronen eindimensional ist, werden wir sie während der Nutzung zu einem Netzwerk verbinden.
        Die ersten drei Neuronen sind Inputs, die folgenden beiden versteckt und das letzte der Output-Knoten.
         */
        public Double predict(List<Double> inputs) {
//            System.out.println("Vorhersage: input1=" + input1 + " input2=" + input2);

            return neurons.get(5).compute(
                    Arrays.asList(
                            neurons.get(4).compute(
                                    Arrays.asList(
                                            neurons.get(2).compute(inputs),
                                            neurons.get(1).compute(inputs))
                            ),
                            neurons.get(3).compute(
                                    Arrays.asList(
                                            neurons.get(1).compute(inputs),
                                            neurons.get(0).compute(inputs))
                            ))
            );
        }

        public void train(List<List<Double>> data, List<Double> answers) {
            Double bestEpochLoss = null;

            // Trainieren so lange, wie Epochen angegeben wurden.
            for (int epoch = 0; epoch < epochs; epoch++) {
                // adapt neuron
                Neuron epochNeuron = neurons.get(epoch % 6);
                epochNeuron.mutate(this.learnFactor);

                List<Double> predictions = new ArrayList<>();
                for (int i = 0; i < data.size(); i++) {
                    predictions.add(i, this.predict(data.get(i)));
                }
                Double thisEpochLoss = Util.meanSquareLoss(answers, predictions);

                if (epoch % 10 == 0) {
                    System.out.println(String.format("Epoch: %s | bestEpochLoss: %.15f | thisEpochLoss: %.15f", epoch, bestEpochLoss, thisEpochLoss));
                }

                if (bestEpochLoss == null) {
                    bestEpochLoss = thisEpochLoss;
                    epochNeuron.remember();
                } else {
                    if (thisEpochLoss < bestEpochLoss) {
                        bestEpochLoss = thisEpochLoss;
                        epochNeuron.remember();
                    } else {
                        epochNeuron.forget();
                    }
                }
            }
        }
    }

    class Neuron {

        private Double bias = ThreadLocalRandom.current().nextDouble(-1, 1);
        private Double oldBias = ThreadLocalRandom.current().nextDouble(-1, 1);
        private final Double[] weights = new Double[MAXINPUTS];
        private final Double[] oldWeights = new Double[MAXINPUTS];

        public Neuron() {
            for (int i = 0; i < MAXINPUTS; i++) {
                weights[i] = ThreadLocalRandom.current().nextDouble(-1, 1);
                oldWeights[i] = ThreadLocalRandom.current().nextDouble(-1, 1);
            }
        }

//        @Override
//        public String toString() {
//            //return String.format("oldBias: %.15f | bias: %.15f | oldWeight1: %.15f | weight1: %.15f | oldWeight2: %.15f | weight2: %.15f", this.oldBias, this.bias, this.oldWeight1, this.weight1, this.oldWeight2, this.weight2);
//            return ".";
//        }
        public void mutate(Double learnFactor) {

            int propertyToChange = ThreadLocalRandom.current().nextInt(0, MAXINPUTS + 1); // +1 wegen Bias
            Double changeFactor = (learnFactor == null) ? ThreadLocalRandom.current().nextDouble(-1, 1) : (learnFactor * ThreadLocalRandom.current().nextDouble(-1, 1));

            if (propertyToChange == MAXINPUTS) {
                this.bias += changeFactor;
            } else {
                this.weights[propertyToChange] += changeFactor;
            }

        }

        public double compute(List<Double> inputs) {

            double preActivation = 0.0;
            // Gewichte * Inputs
            for (int i = 0; i < inputs.size(); i++) {
                preActivation += this.weights[i] * inputs.get(i);
            }
            // Bias hinzufügen
            preActivation += this.bias;

            return Util.sigmoid(preActivation);
        }

        public void forget() {
            bias = oldBias;

            System.arraycopy(oldWeights, 0, weights, 0, MAXINPUTS);

        }

        public void remember() {
            oldBias = bias;

            System.arraycopy(weights, 0, oldWeights, 0, MAXINPUTS);

        }

    }

    static class Util {

        public static double sigmoid(double in) {
            return 1 / (1 + Math.exp(-in));
        }

        public static double sigmoidDeriv(double in) {
            double sigmoid = Util.sigmoid(in);
            return sigmoid * (1 - in);
        }

        public static Double meanSquareLoss(List<Double> correctAnswers, List<Double> predictedAnswers) {
            if (correctAnswers.size() != predictedAnswers.size()) {
                System.err.println("Die Listen sind nicht gleich lang! (meanSquareLoss)");
            }
            double sumSquare = 0;
            for (int i = 0; i < correctAnswers.size(); i++) {
                double error = correctAnswers.get(i) - predictedAnswers.get(i);
                sumSquare += (error * error);
            }
            return sumSquare / (correctAnswers.size());
        }
    }
}
