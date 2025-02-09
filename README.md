# Deep Learning Haskell

Este projeto é uma implementação simples de um modelo de aprendizado profundo em Haskell usando a biblioteca Massiv para manipulação de arrays.

## Estrutura do Projeto

- `app/`: Contém o arquivo `Main.hs` que executa o treinamento do modelo.
- `src/`: Contém a biblioteca principal `Lib.hs` com a definição do modelo e funções de treinamento.
- `test/`: Contém os testes unitários para o modelo.

## Dependências

- `base >= 4.7 && < 5`
- `bytestring`
- `massiv`
- `random`
- `hspec`

## Instalação

Para instalar as dependências e configurar o projeto, execute:

```sh
stack setup
stack build
```

## Execução

Para executar o treinamento do modelo, use:

```sh
stack run
```

## Testes

Para rodar os testes, use:

```sh
stack test
```

## Descrição do Código

### `Lib.hs`

Este módulo contém a definição do modelo, funções de inicialização, forward pass, cálculo de loss e treinamento.

- `Model`: Estrutura de dados que representa o modelo com pesos e bias.
- `initModel`: Função para inicializar o modelo com pesos e bias aleatórios.
- `forward`: Função que realiza o forward pass do modelo.
- `loss`: Função que calcula o erro quadrático médio (MSE) entre as previsões e os valores reais.
- `train`: Função que treina o modelo usando gradiente descendente.

### `Main.hs`

Este módulo executa o treinamento do modelo com dados gerados aleatoriamente.

### `MainSpec.hs`

Este módulo contém testes unitários para verificar a funcionalidade do modelo, incluindo inicialização, forward pass, cálculo de loss e treinamento.

## Licença

Este projeto está licenciado sob a licença BSD-3-Clause. Veja o arquivo `LICENSE` para mais detalhes.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.
