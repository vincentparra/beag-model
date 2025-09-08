package com.rocs.machine.learning.engine;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.local.transforms.AnalyzeLocal;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.EvaluationAveraging;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


import java.io.File;
import java.util.Map;
import java.util.Random;

public class MultiLayerPerceptron {
    private static final int BATCH_SIZE = 20;
    private static final int EPOCH = 300;
    public void multiLayerPerceptronModel() throws Exception {
        Random random = new Random();
        random.setSeed(0xC0FFE);
        FileSplit fileSplit = new FileSplit(new File("src/main/resources/train"),random);
        CSVRecordReader csvReader = new CSVRecordReader(1, ",");
        csvReader.initialize(fileSplit);

        DataAnalysis analysis = AnalyzeLocal.analyze(schema(), csvReader);
        HtmlAnalysis.createHtmlAnalysisFile(analysis,new File("src/main/resources/AI-Human_Generated_Code.html"));

        TransformProcess transformProcess = transformProcess(analysis);
        Schema finalSchema = transformProcess.getFinalSchema();

        TransformProcessRecordReader trainRecordReader = new TransformProcessRecordReader(new CSVRecordReader(1,","),transformProcess);
        trainRecordReader.initialize(fileSplit);

        RecordReaderDataSetIterator trainIterator = new RecordReaderDataSetIterator.Builder(trainRecordReader, BATCH_SIZE)
                .classification(finalSchema.getIndexOfColumn("is_ai_generated"),2)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(multiLayerNetworkConfig(finalSchema));
        model.init();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        uiServer.attach(storage);

        model.addListeners(new StatsListener(storage,50));
        model.fit(trainIterator,EPOCH);

        FileSplit testFile = new FileSplit(new File("src/main/resources/test"));
        TransformProcessRecordReader testRecordReader = new TransformProcessRecordReader(new CSVRecordReader(1,","),transformProcess);
        testRecordReader.initialize(testFile);

        RecordReaderDataSetIterator testIterator = new RecordReaderDataSetIterator.Builder(testRecordReader, BATCH_SIZE)
                .classification(finalSchema.getIndexOfColumn("is_ai_generated"),2)
                .build();

        Evaluation evaluation = model.evaluate(testIterator);
        System.out.println(evaluation.stats());
        System.out.println("MCC: "+evaluation.matthewsCorrelation(EvaluationAveraging.Macro));

        File modelSave = new File("src/main/resources/beag-model-v3.1.bin");
        model.save(modelSave);
        ModelSerializer.addObjectToFile(modelSave, "analysis", analysis.toJson());
        ModelSerializer.addObjectToFile(modelSave, "schema", finalSchema.toJson());
    }

    private Schema schema(){
        Schema schema = new Schema.Builder()
                .addColumnInteger("num_lines")
                .addColumnInteger("num_char")
                .addColumnInteger("num_tokens")
                .addColumnInteger("num_IfStmt")
                .addColumnDouble("ave_tokens_length")
                .addColumnInteger("numMethods")
                .addColumnDouble("avgMethodLength")
                .addColumnInteger("numSwitchStmt")
                .addColumnInteger("numOfLoops")
                .addColumnCategorical("is_ai_generated","0","1")
                .build();
        return schema;
    }
    private TransformProcess transformProcess(DataAnalysis analysis){
        return new TransformProcess.Builder(schema())
                .removeColumns("numSwitchStmt")
                .normalize("num_lines",Normalize.Standardize,analysis)
                .normalize("num_char",Normalize.Standardize, analysis)
                .normalize("num_tokens", Normalize.Standardize, analysis)
                .normalize("num_IfStmt", Normalize.Standardize, analysis)
                .normalize("ave_tokens_length", Normalize.Standardize, analysis)
                .normalize("numMethods", Normalize.Standardize, analysis)
                .normalize("avgMethodLength",Normalize.Standardize,analysis)
                .normalize("numOfLoops",Normalize.Standardize,analysis)
                .build();
    }
    private MultiLayerConfiguration multiLayerNetworkConfig(Schema finalSchema){
        int nIn = 9;
        int nOut = 2;
        int layerSize = 64;
        return new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(0xC0FFE)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam.Builder().learningRate(0.001).build())
                .l2(0.000316)
                .list()
                .layer(new DenseLayer.Builder().nIn(9).nOut(64).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(64).nOut(64).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(64).nOut(2).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.feedForward(finalSchema.numColumns() - 1))
                .build();
    }
}
