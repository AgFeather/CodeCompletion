var Activity = Backbone.Model.extend({
    defaults: {
        name         : null,
        createdAt    : null
    },
    addPayment: function(from, to, amount) {
        var transaction = new Transaction({
            id         : this.collection.getMaxId() + 1,
            activityId : this.id,
            fromId     : from.id,
            toId       : to.id,
            amount     : amount,
        });
        App.transactions.add(transaction);
    }
});

var Person = Backbone.Model.extend({
    defaults: {
        name: null,
        avatar: 'img/person.png'
    }
});

var Transaction = Backbone.Model.extend({
    defaults: {
        fromId : null,
        toId   : null,
        amount : 0
    }
});

var Total = Backbone.Model.extend({
    defaults: {
        person   : null,
        spent    : 0,
        gave     : 0,
        received : 0,
        owes     : false,
        net      : 0
    }
});