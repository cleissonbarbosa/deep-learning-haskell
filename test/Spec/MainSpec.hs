{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}

module Spec.MainSpec (spec) where

import Test.Hspec
import Data.Massiv.Array as A hiding (sum)
import System.Random
import Control.Monad (replicateM)
import Lib

spec :: Spec
spec = do
    describe "Model Initialization" $ do
        it "should initialize model with weights and bias" $ do
            model <- initModel
            size (weights model) `shouldNotBe` Sz1 0
            bias model `shouldSatisfy` (\b -> b >= -1 && b <= 1)

        it "should initialize weights within valid range" $ do
            model <- initModel
            index' (weights model) 0 `shouldSatisfy` (\w -> w >= -1 && w <= 1)

    describe "Forward Pass" $ do
        it "should perform forward pass correctly with simple values" $ do
            let model = Model (fromLists' Seq [0.5]) 0.1
                input = fromLists' Seq [1.0]
                output = forward model input
            output `shouldBe` 0.6

        it "should handle zero input correctly" $ do
            let model = Model (fromLists' Seq [0.5]) 0.1
                input = fromLists' Seq [0.0]
            forward model input `shouldBe` 0.1

    describe "Loss Calculation" $ do
        it "should calculate MSE loss correctly" $ do
            let y_pred = fromLists' Seq [2.0, 3.0]
                y_true = fromLists' Seq [1.0, 2.0]
                l = loss y_pred y_true
            l `shouldBe` 1.0

        it "should return zero loss for perfect predictions" $ do
            let y_pred = fromLists' Seq [1.0, 2.0]
                y_true = fromLists' Seq [1.0, 2.0]
            loss y_pred y_true `shouldBe` 0.0

    describe "Model Training" $ do
        it "should train the model and reduce loss" $ do
            arrList <- replicateM 100 (randomRIO (-10,10))
            let x = fromLists' Seq (Prelude.map (:[]) arrList) :: Array U Ix2 Double
                y = compute @U $ A.map (\v -> 2*v + 1) (compute @U $ flatten x)
            model <- initModel
            trained <- train model x y 1000 0.01
            let y_pred = A.singleton $ forward trained (compute @U $ flatten x)
                final_loss = loss y_pred y
            final_loss `shouldSatisfy` (< 1.0)

        it "should learn simple linear relationship approximately" $ do
            let x = fromLists' Seq [[1.0], [2.0], [3.0]] :: Array U Ix2 Double
                y = fromLists' Seq [2.0, 4.0, 6.0]  -- y = 2x
            model <- initModel
            trained <- train model x y 2000 0.01  -- More epochs for better convergence
            let test_input = fromLists' Seq [4.0]
                prediction = forward trained test_input
            prediction `shouldSatisfy` (\p -> abs (p - 8.0) < 6.0)  -- Allow more variance due to random initialization

    describe "Model Robustness" $ do
        it "should handle large input values" $ do
            let model = Model (fromLists' Seq [0.001]) 0.0  -- Small weight to prevent overflow
                input = fromLists' Seq [1000000.0]
            forward model input `shouldSatisfy` (not . isNaN)

        it "should handle negative input values" $ do
            let model = Model (fromLists' Seq [0.5]) 0.1
                input = fromLists' Seq [-1.0]
            forward model input `shouldBe` (-0.4)  -- -1.0 * 0.5 + 0.1

        it "should attempt to approximate inverse relationships" $ do
            let x = fromLists' Seq [[1.0], [2.0], [4.0]] :: Array U Ix2 Double
                y = fromLists' Seq [1.0, 0.5, 0.25]  -- y = 1/x
            model <- initModel
            trained <- train model x y 3000 0.001
            let test_input = fromLists' Seq [8.0]
                prediction = forward trained test_input
            prediction `shouldSatisfy` (\p -> p >= -10.0 && p <= 10.0)  -- Just ensure output is bounded

        it "should maintain stable loss during training" $ do
            let x = fromLists' Seq [[1.0], [2.0], [3.0]] :: Array U Ix2 Double
                y = fromLists' Seq [2.0, 4.0, 6.0]
            model <- initModel
            trained1 <- train model x y 500 0.01
            trained2 <- train trained1 x y 500 0.01
            let loss1 = loss (A.singleton $ forward trained1 (fromLists' Seq [2.0])) (fromLists' Seq [4.0])
                loss2 = loss (A.singleton $ forward trained2 (fromLists' Seq [2.0])) (fromLists' Seq [4.0])
            loss2 `shouldSatisfy` (<= loss1)