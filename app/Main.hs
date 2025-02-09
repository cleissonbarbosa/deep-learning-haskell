{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}

module Main where

import Lib
import Data.Massiv.Array as A
import System.Random
import Control.Monad (replicateM)

main :: IO ()
main = do
    arrList <- replicateM 100 (randomRIO (-10,10))
    let x = fromLists' Seq (Prelude.map (:[]) arrList) :: Array U Ix2 Double
        y = compute @U $ A.map (\v -> 2*v + 1) (compute @U $ flatten x)
    model <- initModel
    trained <- train model x y 1000 0.01
    putStrLn "Modelo treinado:"
    print trained
