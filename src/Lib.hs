{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}

module Lib
    ( Model(..)
    , initModel
    , forward
    , loss
    , train
    ) where

import Data.Massiv.Array as A hiding (sum, forM_)
import Data.Massiv.Array.Numeric as N
import System.Random
import Control.Monad (forM_, when, replicateM, foldM)
import Prelude hiding (sum)

data Model = Model 
    { weights :: Array U Ix1 Double
    , bias :: Double 
    } deriving Show

-- Helper functions
sum :: (Unbox a, Num a) => Array U Ix1 a -> a 
sum = A.ssum

dot :: Array U Ix1 Double -> Array U Ix1 Double -> Double
dot x y = sum $ compute @U $ A.zipWith (*) x y

forward :: Model -> Array U Ix1 Double -> Double
forward (Model w b) x = dot w x + b

loss :: Array U Ix1 Double -> Array U Ix1 Double -> Double
loss y_pred y_true =
  let diff = compute @U $ A.zipWith (-) y_pred y_true
      squared = compute @U $ A.map (^2) diff
   in sum squared / fromIntegral (unSz $ size squared)

initModel :: IO Model
initModel = do
    randW <- randomRIO (-1,1)
    let w = fromLists' Seq [randW]
    b <- randomRIO (-1,1)
    return $ Model w b

train :: Model -> Array U Ix2 Double -> Array U Ix1 Double -> Int -> Double -> IO Model
train model x y epochs lr = do
    let xFlat = compute @U $ flatten x
    foldM (\m e -> do
        let y_pred = A.singleton $ forward m xFlat
            current_loss = loss y_pred y
            diff = compute @U $ A.zipWith (-) y_pred y
            grad_w = dot (A.singleton $ 2 * sum diff) xFlat
            grad_b = 2 * sum diff
            new_w = fromLists' Seq [index' (weights m) 0 - lr * grad_w]
            new_b = bias m - lr * grad_b
        when (e `mod` 100 == 0) $
            putStrLn $ "Epoch " ++ show e ++ " Loss: " ++ show current_loss
        return $ Model new_w new_b) model [1..epochs]
