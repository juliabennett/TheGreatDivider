
$(document).ready(function(){

	$('#enter').submit(populate);
	$('#dropdown').change(populate);

	function populate() {

		if ($(this).attr('id') == 'enter') {
			var val = $(this).find('input').val();
		}
		else {
			var val = $(this).val();
		}
		
		$('<h2></h2>')
			.html("Conversations about '" + val + "'?")
			.prependTo(".content");

		$('#lefty').empty();
		$('#righty').empty();
		$('#safe').empty();
		$('.toggle').toggleClass('hidden');

		$.getJSON('/search?keyword='  + val, respond);

		return false;
	}

	function respond(res) {
			$('#loading').addClass('hidden');
			$('#again').removeClass('hidden');

			var text = ["<div>Sorry, not enough discussion about this topic.</div>", 
						"<div>Please try again!</div>"].join(" ");

			if (! res['left'] || ! res['right']) {
				$('.clouds').remove();
				$('<h3>' + text + '</h3>')
					.appendTo($('#safe'));
				return false;
			} 

			for (var i=0; i<res['left'].length; i++) {
				draw(res['left'][i], "#lefty", i);
			}

			for (var i=0; i <res['right'].length; i++) { 
				draw(res['right'][i], "#righty", i);
			}

			$(".cloud").click(function(){ return showModal($(this), res);} )
		}	

	function draw(data, side, ind) {
		var words = data['words'],
			  cloud = $("<div class='cloud'></div>"),
			  classes = {
					0: "zero", 
					1: 'one', 
					2: "two", 
					3: 'three', 
					4: 'four', 
			  };

		cloud
			.addClass(classes[ind])
			.appendTo(side);

		for (var i=0; i<words.length; i++) {
			$('<div></div>')
				.html(words[i].text)
				.css('font-size', words[i].size)
				.css('opacity', data['score'])
				.appendTo(cloud);	
		}
	}

	function showModal(current, res) {

		$('#links').empty();
		
		var num = current.attr('class').split(/\s+/)[1], 
				side = current.parent().attr("id"), 
				classes = {
					"zero": 0, 
					'one': 1, 
					"two": 2, 
					'three': 3, 
					'four': 4
				};

		if (side == "lefty") {
			var urls = res['left_urls'];
		}
		if (side == "righty") {
			var urls = res['right_urls'];
		}

		var url_list = urls[classes[num]];

		for (var i=0; i<url_list.length; i++) {

			var div = $("<div></div>")
				.appendTo($('#links'));

			var link = 'https://www.reddit.com/' + url_list[i][0], 
					subred = "/r/" + url_list[i][0].split("/")[2].toLowerCase()
					text = url_list[i][1];

			$('<div> <b>' + subred + '</b> </div>')
				.appendTo(div);

			var block = $('<blockquote></blockquote>')
				.appendTo(div);

			$('<div>' + text + '</div>')
				.appendTo(block);

			$("<a></a>")
				.attr("href", link)
				.html('<button class="btn btn-default read-more" type="submit">read more</button>')
				.appendTo(block);

			$('<br>').appendTo($('#links'));
		}

		$('#myModal').modal();
	}

})

