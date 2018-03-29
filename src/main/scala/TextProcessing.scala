import MlModels.{DecisionTree_Regressor, Linear_Regressor, RandomForest_Regressor}
import TextProcessingController.DataframeCreatorController
import com.sun.javafx.binding.SelectBinding.AsInteger
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.functions.udf
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.collection.mutable
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
//import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.ml.linalg.VectorUDT

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.PCA

object TextProcessing {


  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local[*]")
      .setAppName("Text Mining Project").setSparkHome("src/main/resources")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    val sc = new SparkContext(conf)
    val ss = SparkSession.builder().master("local[*]").appName("Text Mining Project").getOrCreate()
    import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

    val dataframeCreatorController = new DataframeCreatorController()

    /** testDF : result columns : rProductUID , rFilteredWords (all search terms filtered grouped by rProductUID) */
//    val testDF = dataframeCreatorController.getTestDataframe("src/main/resources/test.csv",ss,sc)

//    val trainingDF = dataframeCreatorController.getTrainDataframe("src/main/resources/train.csv",ss,sc)

    val attributesDF = dataframeCreatorController.getAttributesDataframe("src/main/resources/attributes.csv",ss,sc)
    val trainingDF = dataframeCreatorController.getTrainDataframe("src/main/resources/trainSmall.csv",ss,sc)


    /** joinedTextsTrainingAndTestDF contains a dataframe of schema :
      * root
        |-- rID: string (nullable = true)
        |-- rProductUID: string (nullable = true)
        |-- rFilteredWords: array (nullable = true)
        |    |-- element: string (containsNull = true)
        |-- rRelevance: string (nullable = true)
      *
      * At the training process of ml algorithms we will need rFilteredWords, and rRelevance
      *
      * TODO If we find a faster way of executing the following method.. much appreciated
      */
    val joinedTextsTrainingAndTestDF = dataframeCreatorController.uniteTwoDataframeTexts(trainingDF,attributesDF,ss,sc)
//    joinedTextsTrainingAndTestDF.take(5).foreach(println)

    ////TODO creation of labeledVectorTrain file. After first time creation, we just need to load it
//    val idf = dataframeCreatorController.getIDF(trainingDF,ss,sc)
    val idf = dataframeCreatorController.getIDF(joinedTextsTrainingAndTestDF,ss,sc)
    val trainingDfDataInit = idf.select($"rProductUID",$"rFilteredWords", $"rFeatures", $"rRelevance",$"rrLabel").withColumnRenamed("rFeatures", "features").withColumnRenamed("rrLabel", "label")

    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ======================================================= */

    println("Defining features and label for the model...")
    val trainingDfData = trainingDfDataInit.select("features", "label")
    trainingDfData.cache()
    trainingDfData.printSchema()

//    val randomForest_Regressor = new RandomForest_Regressor()
//    randomForest_Regressor.runPrediction(trainingDfData,sc,ss)

//    val linear_Regressor = new Linear_Regressor()
//    linear_Regressor.runPrediction(trainingDfData,sc,ss)

    trainingDfData.printSchema()
    val decisionTree_Regressor = new DecisionTree_Regressor()
    decisionTree_Regressor.runPrediction(trainingDfData,sc,ss)


  }



}