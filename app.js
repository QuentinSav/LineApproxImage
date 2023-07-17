/* Sets the width of the sidebar 
to 250 and the left margin of the 
page content to 250 */
function openNav() {
    document.getElementById(
    "sidebar").style.width = "250px";
    document.getElementById(
    "main").style.marginLeft = "250px";
}

/* Set the width of the sidebar 
to 0 and the left margin of the 
page content to 0 */
function closeNav() {
    document.getElementById(
    "sidebar").style.width = "0";
    document.getElementById(
    "main").style.marginLeft = "0";
}
