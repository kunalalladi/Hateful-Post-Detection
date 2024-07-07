// let chart = bb.generate({
//   data: {
//     columns: [
//       ["Blue", 2],
//       ["orange", 4],
//       ["green", 3],
//     ],
//     type: "donut",
//     onclick: function (d, i) {
//       console.log("onclick", d, i);
//     },
//     onover: function (d, i) {
//       console.log("onover", d, i);
//     },
//     onout: function (d, i) {
//       console.log("onout", d, i);
//     },
//   },
//   donut: {
//     title: "70",
//   },
//   bindto: "#donut-chart",
// });


// var xValues = ["Hateful","Not Hateful"];
// var yValues = [55, 45];
// var barColors = [
//   "#b91d47",

//   "#1e7145"
// ];

// new Chart("myChart", {
//   type: "pie",
//   data: {
//     labels: xValues,
//     datasets: [{
//       backgroundColor: barColors,
//       data: yValues
//     }]
//   },
//   options: {
//     title: {
//       display: true,
//       text: "Hateful Content from this Page"
//     }
//   }
// });