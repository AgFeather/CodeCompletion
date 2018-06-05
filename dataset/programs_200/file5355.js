var ns = namespace('dr.acme.view');

/**
 * Order History Detail View
 * 
 * Will render the model in different ways depending on the function
 * called. They replace the html template in the layout specified
 * with the provided model
 * 
 */
ns.OrderHistoryDetailView =  ns.BaseView.extend({
	init: function() {
		this._super();
		
		this.shippingAddressWidget = new dr.acme.view.AddressWidget("#shippingAddress");
		this.shippingAddressWidget.showTitle(false);
		this.shippingAddressWidget.setLoadingMessage("Loading Shipping Address...");
		this.billingAddressWidget = new dr.acme.view.AddressWidget("#billingAddress");
		this.billingAddressWidget.showTitle(false);
		this.billingAddressWidget.setLoadingMessage("Loading Billing Address...");
	},
    /**
     * Name of the root element for this view
     */
    elementSelector: "#contentArea",
    layoutTemplate: "#orderHistoryDetailTemplate",
    layoutElements:{
    		widgets:[
    		         	{
    		         		containerId:"#orderInfo", 
    	    				message:"Loading Order Info...",
    	    				templateId:"#orderInfo_widget"
    					}, 
    					{
    						containerId:"#paymentInfo", 
    	    				message:"Loading Payment Info...", 
    	    				templateId:"#paymentInfo_widget"
    					}, 
    					{
    						containerId:"#orderProducts", 
    	    				message:"Loading Products...", 
    	    				templateId:"#orderHistoryProducts_widget"
    					}
    					
    		        ]
    			
    },
    /**
     * Events this view dispatches
     */
    events: {
        //This view doesn't have notifications
    },
    /**
     * Handlers for the DOM events must be registered in this method
     */
    initDOMEventHandlers: function() {
        //This view doesn't have notifications
    },
    /**
     * Get the order
     */
    getOrder: function() {
        if(this.modelIsDefined()) {
            return this.model;
        } else {
            return null;
        }
    },
    /**
     * Clear the modes shown in the view
     */
    reset: function() {
    	this._super();
    	this.shippingAddressWidget.reset();
    	this.billingAddressWidget.reset();
    },
    /**
     * Renders loaders for each widget
     */
    renderLoaders:function(){
    	for(var i in this.layoutElements.widgets) {
    		this.showLoaderOnComponent(this.layoutElements.widgets[i].containerId, 
    				this.layoutElements.widgets[i].message);
    	}
    	this.shippingAddressWidget.render(true);
    	this.billingAddressWidget.render(true);
    },
    /**
     * Renders the order info with the model given as parameter
     */
    renderOrderInfo :function(model){
    	for(var i in this.layoutElements.widgets) {
    		this.applyTemplate(this.layoutElements.widgets[i].containerId,
    				this.layoutElements.widgets[i].templateId , model);    		
    	} 
    	this.renderAddresses(model);   	   
    },
    
    /**
     * Renders Shipping and Billing Addresses
     */
    renderAddresses:function(model){
    	$("#shippingAddress").html("<h3>Shipping Address</h3>");
		this.shippingAddressWidget.setAddress(model.shippingAddress);
		this.shippingAddressWidget.render(true);
		
		$("#billingAddress").html("<h3>Billing Address</h3>");
		this.billingAddressWidget.setAddress(model.billingAddress);
		this.billingAddressWidget.render(true);
    },
	/**
	 * Order detail rendering
	 */
	setOrder: function(model) {
	    this.model = model;
	}
});