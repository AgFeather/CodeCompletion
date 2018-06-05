// tableit.js: Simple responsive tables
// MIT license http://www.opensource.org/licenses/mit-license.php/
// Based on method developed by Chris Coyier http://css-tricks.com/responsive-data-tables/
// @author Joshua McGrath http://designjem.com

 (function( $ ){

  $.fn.tableit = function(options) {

	var settings = $.extend( {
		firstRowHeading:		false,
		evenOdd:				true,
		headingMaxCharacters:	null
	}, options);

	return this.each(function() {
		var t = $(this);

		t.addClass('tableit');

		var theadExists = false;
		if (t.find('thead').length>0){
			theadExists = true;
		}

		if(settings.firstRowHeading === true && theadExists === false){
			t.addClass('noHead');
			t.find('tr:not(:first)').each(function(){
				$(this).children('td').each(function(index){
					heading = t.find('tr:first').children('td:eq('+index+')').text();
					$(this).attr('data-title',heading);
					if(settings.headingMaxCharacters !== null && settings.headingMaxCharacters !== 0){
						var copy = $(this).text().length;
						//console.log(heading + ": " + copy);
						if(heading.length > settings.headingMaxCharacters){
							$(this).addClass('overflow');
						}
					}
				});
			});
		}
		else{
			t.find('tbody tr').each(function(){
				$(this).children('td').each(function(index){
					if(t.find('thead tr').children('th').length > 0){
						heading = t.find('thead tr').children('th:eq('+index+')').text();
					}else{
						heading = t.find('thead tr').children('td:eq('+index+')').text();
					}
					$(this).attr('data-title',heading);
					if(settings.headingMaxCharacters !== null && settings.headingMaxCharacters !== 0){
						if(heading.length > settings.headingMaxCharacters){
							$(this).addClass('overflow');
						}
					}
				});
			});
		}

		if(settings.evenOdd){
			t.find('tr:odd').addClass('odd');
		}
	});

  };

})( jQuery );
