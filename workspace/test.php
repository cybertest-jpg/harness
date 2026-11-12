<?php

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['data']) && $_POST['data'] === '12345') {
    echo 'ok';
}
