package com.rocs.machine.learning;

import com.rocs.machine.learning.engine.MultiLayerPerceptron;

public class BeagPluginModel {
    public static void main(String[] args) {
        MultiLayerPerceptron multiLayerPerceptron = new MultiLayerPerceptron();
        try {
            multiLayerPerceptron.multiLayerPerceptronModel();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
