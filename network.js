
console.log("In principio erat Verbum!\n\n");

class Network {
    constructor () {
        this.learningRate = 0.1;        // Taxa de aprendizado.
        this.limit  = 10000;            // Limite das iterações de aprendizado.
        this.input  = new Layer(2, 0);  // Nodes por camada, e total de pesos por node.
        this.hidden = new Layer(5, 2);
        this.output = new Layer(1, 5);
    }

    // Ativação.
    sigmoid (x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Gradient.
    dsigmoid (x) {
        return x * (1-x); 
    }
    
    // Realiza raciocínio - Feedforward.
    process (option1, option2) {

        // Inicializa camada de entrada.
        this.input.nodes[0].output = option1;
        this.input.nodes[1].output = option2;

        // Calcula saída, input -> hidden.
        for (var a=0; a<this.hidden.nodes.length; a++) {
            var total = 0;

            for (var b=0; b<this.input.nodes.length; b++) 
                total += parseFloat( this.input.nodes[b].output * this.hidden.nodes[a].weight[b] );
            
            this.hidden.nodes[a].output = this.sigmoid(total);
        }

        // Calcula saída, hidden -> output.
        for (var a=0; a<this.output.nodes.length; a++) {
            var total = 0;

            for (var b=0; b<this.hidden.nodes.length; b++) 
                total += parseFloat( this.hidden.nodes[b].output * this.output.nodes[a].weight[b] );
            
            this.output.nodes[a].output = this.sigmoid(total);
        }
    }

    // Realiza backpropagation.
    backPropagation (expectedValue) {

        //
        // Calcula erros da rede.
        //

        // Calcula erro da camada de saída.
        for (var a=0; a<this.output.nodes.length; a++)
            this.output.nodes[a].error = expectedValue - this.output.nodes[a].output;

        // Calcula erro da camada oculta.
        for (var a=0; a<this.hidden.nodes.length; a++) {
            var total = 0;

            for (var b=0; b<this.output.nodes.length; b++) 
                total += parseFloat( this.output.nodes[b].error * this.output.nodes[b].weight[a] );
            
            this.hidden.nodes[a].error = total;
        }

        //
        // Atualiza valores dos pesos (gradient).
        //

        // Camada, input -> hidden.
        for (var a=0; a<this.hidden.nodes.length; a++) {
            for (var b=0; b<this.input.nodes.length; b++) {
                this.hidden.nodes[a].weight[b] += parseFloat( 
                    this.learningRate *                          // Learning rate.
                    this.hidden.nodes[a].error *                 // Layer error.
                    this.input.nodes[b].output *                 // Sigmoid  | (1.0/(1.0+exp(-X)))
                    this.dsigmoid( this.hidden.nodes[a].output ) // Dsigmoid | (X*(1.0-X))
                );
            }
        }
        
        // Camada, hidden -> output.
        for (var a=0; a<this.output.nodes.length; a++) {
            for (var b=0; b<this.hidden.nodes.length; b++) {
                this.output.nodes[a].weight[b] += parseFloat( 
                    this.learningRate *
                    this.output.nodes[a].error *
                    this.hidden.nodes[b].output *
                    this.dsigmoid( this.output.nodes[a].output ) 
                );
            }
        }
    }

    // Realiza treinamento da rede.
    train (dataset) {
        for (var n=0; n<this.limit; n++) {
            for (var a=0; a<dataset.input.length; a++) {

                // Feedforward.
                this.process(dataset.input[a][0], dataset.input[a][1]);

                // Backpropagation.
                this.backPropagation(dataset.output[a]);
            }
        }
    }

    // Realiza previsão.
    predict (param) {
        this.process(param[0], param[1]);

        var responseValue = this.output.nodes[0].output;
        console.log('input: '+ param[0] +','+ param[1] +' -> '+ Math.round(responseValue) +' | '+ responseValue);
    }
}

class Layer {
    constructor (totalNodes, totalWeights) {
        this.nodes = [];
        for (var a=0; a<totalNodes; a++)
            this.nodes.push(new Node(totalWeights));
    }
}

class Node {
    constructor (totalWeights) {
        this.weight = [];
        this.error  = 0;
        this.output = 1;

        // Inicializa pesos.
        for (var a=0; a<totalWeights; a++)
            this.weight.push(Math.random() * 2 - 1);
    }
}

// Inicia rede e realiza treinamento.
var VerbumNetwork = new Network();

var dataset = {
    input: [
        [1,1], [1,0], [0,0], [0,1]
    ],
    output: [
        0, 1, 0, 1
    ]
};

VerbumNetwork.train(dataset);

// Realiza previsão.
for (var a=0; a<dataset.input.length; a++)
    VerbumNetwork.predict(dataset.input[a]);

console.log(VerbumNetwork)


