# criptowords

Esse programa está em testes, para compilar basta entrar na pasta ``/src/`` depois compilar com: 

``g++ -O3 -mavx2 main.cpp cli.cpp brute_engine.cpp cli_parser.cpp -o runner -lsecp256k1 -lcrypto -lOpenCL -lpthread -std=c++26 -march=native``

apos isso, o executavel será gerado com o nome ``runner``  em ``/src/`` 

por estar em teste, pode ser que ocorra erros na compilação ou na execução do programa, se ocorrer, me informe via as issues para que as correções sejam realizadas.
